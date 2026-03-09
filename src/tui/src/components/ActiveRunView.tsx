import { useState, useEffect, useCallback, useRef } from "react"
import { useRenderer, useTerminalDimensions } from "@opentui/react"
import { BrailleProgressBar } from "./BrailleProgressBar"
import { ImagePreview } from "./ImagePreview"
import { StatusBar } from "./StatusBar"
import { RuneSpinner } from "./RuneSpinner"
import { RuneRevealMatch } from "./RuneRevealMatch"
import { AnsiArt } from "./AnsiArt"
import type { AnsiSegment } from "../utils/ansiArt"
import type { ScanEvent, LogEntry, BoxInfo, ResumeStats, RunMode } from "../types"
import { colors, confidenceColor } from "../theme"
import {
  bindMetadataToTsv,
  loadRunMetadataById,
  makeRunMetadata,
  saveRunMetadata,
  withProgress,
  withStatus,
  type ResumeProgress,
  type RunMetadataRecord,
  type RunStatus,
} from "../utils/runMetadata"

interface ActiveRunViewProps {
  sendCommand: (cmd: any) => void
  onEvent: (listener: (event: ScanEvent) => void) => () => void
  onComplete: () => void
  onQuit: () => void
  resume?: boolean
  autoStart?: boolean
  outputTsvPath?: string
  inputRootDir?: string
  runMode?: RunMode
  singleBoxDir?: string
  resumeStats?: ResumeStats | null
  eyeLogoRows?: AnsiSegment[][]
  textLogoRaw?: string
}

type RunPhase = "pre_scan" | "ready" | "running" | "paused" | "complete"
type PreviewMode = "live" | "browse"
type MatchSource = "GCD" | "ComicVine" | "Search"

interface RunResultRow {
  key: string
  title: string
  issue_number: string
  month: string
  year: string
  publisher: string
  box: string
  filename: string
  confidence: string
  imagePath: string | null
}

interface SearchFields {
  title: string
  issue: string
  month: string
  year: string
  publisher: string
}

interface LatestMatch {
  title: string
  issue: string
  month: string
  year: string
  publisher: string
  confidence: string
  source: MatchSource
  foundAtMs: number
}

function isConfidenceLevel(value: string): value is "high" | "medium" | "low" {
  return value === "high" || value === "medium" || value === "low"
}

function confidenceBackground(level: string): string {
  if (level === "high") return "#0f3d0f"
  if (level === "medium") return "#5a3a12"
  if (level === "low") return "#4a1212"
  return colors.muted
}

function clip(value: string, width: number): string {
  if (width <= 0) return ""
  if (value.length <= width) return value.padEnd(width)
  if (width <= 3) return value.slice(0, width)
  return value.slice(0, width - 3) + "..."
}

function truncate(value: string, width: number): string {
  if (width <= 0) return ""
  if (value.length <= width) return value
  if (width <= 3) return value.slice(0, width)
  return value.slice(0, width - 3) + "..."
}

function formatSearchFields(fields: SearchFields): string {
  const parts: string[] = []
  if (fields.title) parts.push(fields.title)
  if (fields.issue) parts.push(`#${fields.issue}`)
  if (fields.month) parts.push(fields.month.toUpperCase())
  if (fields.year) parts.push(fields.year)
  if (fields.publisher) parts.push(`(${fields.publisher})`)
  return parts.join(" ").trim() || "current image"
}

function formatHeaderStatus(text: string): string {
  const trimmed = text.trim()
  if (!trimmed) return ""
  const withoutTrailingPunctuation = trimmed.replace(/(?:\.\.\.|…|\.)\s*$/, "")
  return `${withoutTrailingPunctuation}...`
}

function statusBarRows(left: string, right: string, terminalWidth: number): number {
  const horizontalPadding = 2
  const contentWidth = Math.max(0, terminalWidth - (horizontalPadding * 2))
  const hasBoth = left.length > 0 && right.length > 0
  const minGap = hasBoth ? 2 : 0
  const fitsSingleLine = !hasBoth || (left.length + right.length + minGap <= contentWidth)
  return fitsSingleLine ? 1 : 2
}

function buildResultsFooterLayout(
  countText: string,
  stats: { high: number; medium: number; low: number },
  width: number,
): {
  count: string
  gap: string
  high: string | null
  medium: string | null
  low: string | null
  separator: string
} {
  if (width <= 0) {
    return { count: "", gap: "", high: null, medium: null, low: null, separator: " " }
  }

  const variants = [
    {
      separator: "  ",
      high: `high:${stats.high}`,
      medium: `medium:${stats.medium}`,
      low: `low:${stats.low}`,
    },
    {
      separator: " ",
      high: `h:${stats.high}`,
      medium: `m:${stats.medium}`,
      low: `l:${stats.low}`,
    },
  ] as const

  for (const variant of variants) {
    const combos = [
      { high: variant.high, medium: variant.medium, low: variant.low },
      { high: variant.high, medium: variant.medium, low: null },
      { high: variant.high, medium: null, low: null },
      { high: null, medium: null, low: null },
    ] as const

    for (const combo of combos) {
      const segments: string[] = []
      for (const value of [combo.high, combo.medium, combo.low]) {
        if (value !== null) {
          segments.push(value)
        }
      }
      const statsWidth = segments.length === 0
        ? 0
        : segments.reduce((sum, segment) => sum + segment.length, 0) + (variant.separator.length * (segments.length - 1))
      const minGap = segments.length > 0 ? 2 : 0
      const maxCountWidth = width - statsWidth - minGap
      if (maxCountWidth <= 0) continue
      const count = truncate(countText, maxCountWidth)
      const gapWidth = Math.max(minGap, width - count.length - statsWidth)
      return {
        count,
        gap: " ".repeat(gapWidth),
        high: combo.high,
        medium: combo.medium,
        low: combo.low,
        separator: variant.separator,
      }
    }
  }

  return {
    count: truncate(countText, width),
    gap: "",
    high: null,
    medium: null,
    low: null,
    separator: " ",
  }
}

// Pipeline step constants
const STEP_COLORS_KEYS = ["cyan", "yellow", "magenta", "magenta"] as const
const STEP_IDX = { extract: 0, gcd: 1, comicvine: 2, compare: 3 } as const

function fieldIndexMap(header: string[]): Record<string, number> {
  const map: Record<string, number> = {}
  for (let i = 0; i < header.length; i += 1) {
    map[header[i]] = i
  }
  return map
}

function buildProcessedByBox(rows: RunResultRow[]): Record<string, number> {
  const counts: Record<string, number> = {}
  for (const row of rows) {
    if (!row.box || !row.filename) continue
    counts[row.box] = (counts[row.box] ?? 0) + 1
  }
  return counts
}

function buildResumeBoxSnapshot(
  previewBoxes: BoxInfo[],
  processedByBox: Record<string, number>,
): { currentBox: string; currentBoxIndex: number; boxTotal: number; done: number } | null {
  if (previewBoxes.length === 0) return null

  let focusIndex = previewBoxes.length - 1
  for (let i = 0; i < previewBoxes.length; i += 1) {
    const box = previewBoxes[i]
    const processed = Math.min(processedByBox[box.name] ?? 0, box.count)
    if (processed < box.count) {
      focusIndex = i
      break
    }
  }

  const focusBox = previewBoxes[focusIndex]
  const done = Math.min(processedByBox[focusBox.name] ?? 0, focusBox.count)

  return {
    currentBox: focusBox.name,
    currentBoxIndex: focusIndex + 1,
    boxTotal: focusBox.count,
    done,
  }
}

export function ActiveRunView({
  sendCommand,
  onEvent,
  onComplete,
  onQuit,
  resume = false,
  autoStart = false,
  outputTsvPath = "",
  inputRootDir = "",
  runMode = "batch",
  singleBoxDir = "",
  resumeStats = null,
  eyeLogoRows,
  textLogoRaw,
}: ActiveRunViewProps) {
  const renderer = useRenderer()
  const { width, height } = useTerminalDimensions()

  const [phase, setPhase] = useState<RunPhase>("pre_scan")
  const [previewMode, setPreviewMode] = useState<PreviewMode>("live")
  const [previewBoxes, setPreviewBoxes] = useState<BoxInfo[]>([])
  const [totalImages, setTotalImages] = useState(0)
  const [globalProcessed, setGlobalProcessed] = useState(resume ? (resumeStats?.processedImages ?? 0) : 0)
  const [globalTotal, setGlobalTotal] = useState(0)

  const [currentBox, setCurrentBox] = useState("")
  const [currentBoxIndex, setCurrentBoxIndex] = useState(0)
  const [totalBoxes, setTotalBoxes] = useState(0)
  const [done, setDone] = useState(0)
  const [boxTotal, setBoxTotal] = useState(0)
  const [currentImagePath, setCurrentImagePath] = useState<string | null>(null)
  const [runResults, setRunResults] = useState<RunResultRow[]>([])
  const [selectedResultIndex, setSelectedResultIndex] = useState(-1)
  const [logEntries, setLogEntries] = useState<LogEntry[]>([])
  const [stats, setStats] = useState({ high: 0, medium: 0, low: 0 })
  const [escPromptActive, setEscPromptActive] = useState(false)
  const [resumeRowsReady, setResumeRowsReady] = useState(!resume)
  const [metadataReady, setMetadataReady] = useState(false)
  const [latestMatch, setLatestMatch] = useState<LatestMatch | null>(null)
  const [liveStatus, setLiveStatus] = useState<{ text: string; color: string }>({
    text: formatHeaderStatus("Waiting for Scanner Events"),
    color: colors.dimmed,
  })

  // Pipeline step colors resolved from theme
  const STEP_COLORS = STEP_COLORS_KEYS.map(k => colors[k])

  const [pipeline, setPipeline] = useState<{
    filename: string
    activeStep: number      // -1 = idle
    visited: boolean[]      // tracks which steps were entered
    statusText: string
    statusColor: string
    confidence: string      // set on ScanComplete
  }>({
    filename: "",
    activeStep: -1,
    visited: [false, false, false, false],
    statusText: "",
    statusColor: colors.dimmed,
    confidence: "",
  })

  // Refs to avoid stale closures in useKeyboard
  const phaseRef = useRef<RunPhase>("pre_scan")
  const previewModeRef = useRef<PreviewMode>("live")
  const previewBoxesRef = useRef<BoxInfo[]>([])
  const activeBoxRef = useRef("")
  const currentImagePathRef = useRef<string | null>(null)
  const runResultsRef = useRef<RunResultRow[]>([])
  const processedByBoxRef = useRef<Record<string, number>>(resume && resumeStats ? { ...resumeStats.processedByBox } : {})
  const imagePathByResultKeyRef = useRef<Map<string, string>>(new Map())
  const searchFieldsRef = useRef<SearchFields>({
    title: "",
    issue: "",
    month: "",
    year: "",
    publisher: "",
  })
  const pipelineRef = useRef(pipeline)
  const matchSourceForImageRef = useRef<MatchSource | null>(null)
  const resumeLoadedRef = useRef(false)
  const resumeRowsReadyRef = useRef(!resume)
  const autoStartTriggeredRef = useRef(false)
  const metadataReadyRef = useRef(false)
  const runIdRef = useRef<string | null>(resumeStats?.runId ?? null)
  const metadataRef = useRef<RunMetadataRecord | null>(null)
  const totalImagesRef = useRef(0)
  const globalTotalRef = useRef(0)
  const escPressedRef = useRef(false)
  const escTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Keep refs in sync with state
  useEffect(() => { phaseRef.current = phase }, [phase])
  useEffect(() => { previewModeRef.current = previewMode }, [previewMode])
  useEffect(() => { previewBoxesRef.current = previewBoxes }, [previewBoxes])
  useEffect(() => { currentImagePathRef.current = currentImagePath }, [currentImagePath])
  useEffect(() => { runResultsRef.current = runResults }, [runResults])
  useEffect(() => { pipelineRef.current = pipeline }, [pipeline])
  useEffect(() => { resumeRowsReadyRef.current = resumeRowsReady }, [resumeRowsReady])
  useEffect(() => { metadataReadyRef.current = metadataReady }, [metadataReady])
  useEffect(() => { totalImagesRef.current = totalImages }, [totalImages])
  useEffect(() => { globalTotalRef.current = globalTotal }, [globalTotal])

  const addLog = useCallback((tag: string, message: string, color: string) => {
    setLogEntries(prev => {
      const next = [...prev, { tag, message, color }]
      return next.length > 200 ? next.slice(next.length - 200) : next
    })
  }, [])

  const updateLiveStatus = useCallback((text: string, color: string = colors.accent) => {
    setLiveStatus({ text: formatHeaderStatus(text), color })
  }, [])

  const buildProgressSnapshot = useCallback((): ResumeProgress => {
    const processedByBox = { ...processedByBoxRef.current }
    const uniqueBoxes = Object.keys(processedByBox)
      .filter((box) => (processedByBox[box] ?? 0) > 0)
      .sort()
    const processedImages = uniqueBoxes.reduce((sum, box) => sum + (processedByBox[box] ?? 0), 0)
    const completedBoxes = previewBoxesRef.current
      .filter((box) => (processedByBox[box.name] ?? 0) >= box.count && box.count > 0)
      .map((box) => box.name)
      .sort()

    return {
      processedImages,
      processedBoxCount: uniqueBoxes.length,
      processedByBox,
      uniqueBoxes,
      completedBoxes,
    }
  }, [])

  const persistMetadataFast = useCallback((status?: RunStatus) => {
    if (!metadataRef.current) return
    let updated = withProgress(metadataRef.current, buildProgressSnapshot())
    if (status) updated = withStatus(updated, status)
    metadataRef.current = updated
    saveRunMetadata(updated)
  }, [buildProgressSnapshot])

  const refreshMetadataIdentity = useCallback(async (status?: RunStatus) => {
    if (!metadataRef.current || !outputTsvPath) return
    let updated = withProgress(metadataRef.current, buildProgressSnapshot())
    if (status) updated = withStatus(updated, status)
    updated = await bindMetadataToTsv(updated, outputTsvPath)
    metadataRef.current = updated
    saveRunMetadata(updated)
  }, [buildProgressSnapshot, outputTsvPath])

  // Send scan-preview on mount
  useEffect(() => {
    sendCommand({ cmd: "scan-preview" })
  }, [sendCommand])

  useEffect(() => {
    if (metadataReady) return
    if (previewBoxes.length === 0) return
    if (resume && !resumeRowsReady) return

    let cancelled = false
    const initMetadata = async () => {
      try {
        if (!outputTsvPath.trim()) {
          if (!cancelled) setMetadataReady(true)
          return
        }

        const boxImageTotals: Record<string, number> = {}
        const boxesInScope = previewBoxes.map((box) => box.name)
        const totalImagesInScope = totalImagesRef.current > 0
          ? totalImagesRef.current
          : previewBoxes.reduce((sum, box) => sum + box.count, 0)
        for (const box of previewBoxes) {
          boxImageTotals[box.name] = box.count
        }

        const progress = buildProgressSnapshot()
        let record = runIdRef.current ? loadRunMetadataById(runIdRef.current) : null

        if (record) {
          record = withProgress(record, progress)
          record = withStatus(record, "running")
          record = await bindMetadataToTsv(record, outputTsvPath)
        } else {
          record = await makeRunMetadata({
            runId: runIdRef.current ?? undefined,
            status: "running",
            runMode,
            inputRootDir,
            singleBoxDir,
            boxesInScope,
            totalBoxes: previewBoxes.length,
            totalImages: totalImagesInScope,
            boxImageTotals,
            progress,
            tsvPath: outputTsvPath,
          })
          runIdRef.current = record.run_id
        }

        if (cancelled) return
        metadataRef.current = record
        saveRunMetadata(record)
      } catch {
        // Metadata failures should not block scanning.
      } finally {
        if (!cancelled) setMetadataReady(true)
      }
    }

    void initMetadata()
    return () => {
      cancelled = true
    }
  }, [
    metadataReady,
    previewBoxes,
    resume,
    resumeRowsReady,
    outputTsvPath,
    buildProgressSnapshot,
    runMode,
    inputRootDir,
    singleBoxDir,
  ])

  useEffect(() => {
    if (!resume || !outputTsvPath || resumeLoadedRef.current) return
    resumeLoadedRef.current = true

    let cancelled = false
    const loadResumeRows = async () => {
      try {
        const file = Bun.file(outputTsvPath)
        if (!(await file.exists())) {
          updateLiveStatus(`Resume Mode: TSV Not Found at ${outputTsvPath}`, colors.yellow)
          return
        }

        const raw = await file.text()
        const lines = raw.split(/\r?\n/).filter(line => line.trim().length > 0)
        if (lines.length <= 1) {
          updateLiveStatus(`Resume Mode: TSV Has No Previous Rows (${outputTsvPath})`, colors.yellow)
          return
        }

        const header = lines[0].split("\t").map(h => h.trim())
        const idx = fieldIndexMap(header)
        const rootDir = inputRootDir.replace(/[\\/]+$/, "")
        const loaded: RunResultRow[] = []
        const counts = { high: 0, medium: 0, low: 0 }
        const loadedProcessedByBox: Record<string, number> = {}

        for (let i = 1; i < lines.length; i += 1) {
          const fields = lines[i].split("\t")
          const box = fields[idx.box] ?? ""
          const filename = fields[idx.filename] ?? ""
          const confidence = ((fields[idx.confidence] ?? "").trim().toLowerCase()) || "low"
          const key = `${box}::${filename || `row-${i}`}`
          const imagePath = rootDir && box && filename ? `${rootDir}/${box}/${filename}` : null

          loaded.push({
            key,
            title: fields[idx.title] ?? "",
            issue_number: fields[idx.issue_number] ?? "",
            month: fields[idx.month] ?? "",
            year: fields[idx.year] ?? "",
            publisher: fields[idx.publisher] ?? "",
            box,
            filename,
            confidence,
            imagePath,
          })

          if (isConfidenceLevel(confidence)) {
            counts[confidence] += 1
          }
          if (box && filename) {
            loadedProcessedByBox[box] = (loadedProcessedByBox[box] ?? 0) + 1
          }
        }

        if (cancelled) return
        processedByBoxRef.current = loadedProcessedByBox
        setRunResults(loaded)
        setSelectedResultIndex(loaded.length > 0 ? loaded.length - 1 : -1)
        setStats(counts)
        const processedCount = Object.values(loadedProcessedByBox).reduce((sum, count) => sum + count, 0)
        setGlobalProcessed(Math.min(processedCount, globalTotalRef.current || processedCount))
        addLog("SYSTEM", `Loaded ${loaded.length} existing TSV result rows`, colors.green)
        updateLiveStatus(`Loaded ${loaded.length} Existing Results from TSV`, colors.green)
      } catch (error: any) {
        if (cancelled) return
        const message = error?.message ?? String(error)
        addLog("ERROR", `Failed to load TSV resume rows: ${message}`, colors.red)
        updateLiveStatus(`Failed to Load TSV Resume Rows: ${message}`, colors.red)
      } finally {
        if (!cancelled) {
          setResumeRowsReady(true)
        }
      }
    }

    loadResumeRows()
    return () => {
      cancelled = true
    }
  }, [resume, outputTsvPath, inputRootDir, addLog, updateLiveStatus])

  useEffect(() => {
    if (!autoStart || autoStartTriggeredRef.current || previewBoxes.length === 0) return
    if (resume && !resumeRowsReady) return
    if (!metadataReady) return

    autoStartTriggeredRef.current = true
    setPhase("running")
    addLog("SYSTEM", `Found ${totalImages} images in ${previewBoxes.length} box(es) — starting run`, colors.green)
    updateLiveStatus(`Starting Run: ${totalImages} Images Across ${previewBoxes.length} Box(es)`, colors.green)
    sendCommand({ cmd: "scan", resume })
  }, [autoStart, previewBoxes, totalImages, resume, resumeRowsReady, metadataReady, addLog, updateLiveStatus, sendCommand])

  useEffect(() => {
    if (!resume || previewBoxes.length === 0 || currentBox) return
    if (Object.keys(processedByBoxRef.current).length === 0) return

    const snapshot = buildResumeBoxSnapshot(previewBoxes, processedByBoxRef.current)
    if (!snapshot) return

    setCurrentBox(snapshot.currentBox)
    activeBoxRef.current = snapshot.currentBox
    setCurrentBoxIndex(snapshot.currentBoxIndex)
    setBoxTotal(snapshot.boxTotal)
    setDone(snapshot.done)
  }, [resume, previewBoxes, currentBox])

  // Listen for events
  useEffect(() => {
    return onEvent((event) => {
      switch (event.event) {
        case "scan_preview":
          setPreviewBoxes(event.boxes)
          setTotalImages(event.total_images)
          setTotalBoxes(event.boxes.length)
          if (event.boxes.length === 0) {
            setMetadataReady(true)
          }
          setGlobalTotal(event.total_images)
          if (resume) {
            const seededProcessed = Object.values(processedByBoxRef.current).reduce((sum, count) => sum + count, 0)
            setGlobalProcessed(Math.min(seededProcessed, event.total_images))
          } else {
            processedByBoxRef.current = {}
            setGlobalProcessed(0)
          }
          if (!autoStart) {
            setPhase("ready")
            addLog("SYSTEM", `Found ${event.total_images} images in ${event.boxes.length} box(es) — [Enter] Start`, colors.green)
            updateLiveStatus(`Ready: ${event.total_images} Images Across ${event.boxes.length} Box(es)`, colors.accent)
          }
          break

        case "scan_started":
          addLog("SYSTEM", `Backend output TSV: ${event.output_path || "(none)"}`, colors.dimmed)
          updateLiveStatus(`Run Started. Output TSV: ${event.output_path || "(none)"}`, colors.dimmed)
          break

        case "BoxStarted":
          const processedByBox = { ...processedByBoxRef.current, ...buildProcessedByBox(runResultsRef.current) }
          const resumedDoneForBox = Math.min(processedByBox[event.box_name] ?? 0, (
            previewBoxesRef.current.find((box) => box.name === event.box_name)?.count
            ?? processedByBox[event.box_name]
            ?? 0
          ) + event.image_count)
          const currentBoxInfo = previewBoxesRef.current.find((box) => box.name === event.box_name)
          const totalForBox = currentBoxInfo?.count ?? (event.image_count + resumedDoneForBox)
          const boxIndex = previewBoxesRef.current.findIndex((box) => box.name === event.box_name)

          setCurrentBox(event.box_name)
          activeBoxRef.current = event.box_name
          setCurrentBoxIndex(boxIndex >= 0 ? boxIndex + 1 : 1)
          setBoxTotal(totalForBox)
          setDone(resumedDoneForBox)
          if (resume && resumedDoneForBox > 0) {
            addLog("BOX", `Continuing ${event.box_name} (${resumedDoneForBox}/${totalForBox} processed, ${event.image_count} remaining)`, colors.accent)
            updateLiveStatus(`Processing ${event.box_name} (${resumedDoneForBox}/${totalForBox} Processed)`, colors.accent)
          } else {
            addLog("BOX", `Starting ${event.box_name} (${event.image_count} images)`, colors.accent)
            updateLiveStatus(`Processing ${event.box_name} (${event.image_count} Images)`, colors.accent)
          }
          break

        case "ImageLoading":
          matchSourceForImageRef.current = null
          if ("image_path" in event && event.image_path) {
            setCurrentImagePath(event.image_path)
            currentImagePathRef.current = event.image_path
            const key = `${activeBoxRef.current}::${event.filename}`
            imagePathByResultKeyRef.current.set(key, event.image_path)
          }
          // Reset pipeline for new image
          setPipeline({
            filename: event.filename,
            activeStep: STEP_IDX.extract,
            visited: [true, false, false, false],
            statusText: `Loading ${event.filename}`,
            statusColor: colors.cyan,
            confidence: "",
          })
          addLog("VLM", event.filename, colors.cyan)
          updateLiveStatus(`Loading ${event.filename}`, colors.dimmed)
          break

        case "VLMExtracting":
          setPipeline(prev => ({
            ...prev,
            activeStep: STEP_IDX.extract,
            visited: [true, ...prev.visited.slice(1)] as boolean[],
            statusText: "Extracting Cover Details",
            statusColor: colors.cyan,
          }))
          addLog("VLM", "Extracting cover details...", colors.cyan)
          updateLiveStatus("Extracting Cover Details", colors.cyan)
          break

        case "VLMResult":
          searchFieldsRef.current = {
            ...searchFieldsRef.current,
            title: event.title || searchFieldsRef.current.title,
            issue: event.issue || searchFieldsRef.current.issue,
            publisher: event.publisher || searchFieldsRef.current.publisher,
            year: event.year || searchFieldsRef.current.year,
          }
          setPipeline(prev => ({
            ...prev,
            statusText: `Found: ${event.title} #${event.issue} (${event.publisher})`,
            statusColor: colors.cyan,
          }))
          addLog("VLM", `${event.title} #${event.issue} (${event.publisher})`, colors.cyan)
          break

        case "GCDSearching":
          searchFieldsRef.current = {
            ...searchFieldsRef.current,
            title: event.title || searchFieldsRef.current.title,
            issue: event.issue || searchFieldsRef.current.issue,
          }
          setPipeline(prev => {
            const visited = [...prev.visited]
            visited[STEP_IDX.gcd] = true
            return {
              ...prev,
              activeStep: STEP_IDX.gcd,
              visited: visited as boolean[],
              statusText: `Searching GCD (Strategy ${event.strategy}) for ${formatSearchFields(searchFieldsRef.current)}`,
              statusColor: colors.yellow,
            }
          })
          updateLiveStatus(`Searching GCD (Strategy ${event.strategy}) for ${formatSearchFields(searchFieldsRef.current)}`, colors.yellow)
          addLog("GCD", `Strategy ${event.strategy}: "${event.title}" #${event.issue}`, colors.yellow)
          break

        case "GCDMatchFound":
          matchSourceForImageRef.current = "GCD"
          setPipeline(prev => ({
            ...prev,
            statusText: `GCD Candidate: ${event.title} (Score: ${event.confidence.toFixed(1)})`,
            statusColor: colors.yellow,
          }))
          addLog("GCD", `Match: ${event.title} (${event.confidence.toFixed(1)})`, colors.yellow)
          updateLiveStatus(`GCD Candidate Found: ${event.title}`, colors.yellow)
          break

        case "GCDNoMatch":
          setPipeline(prev => ({
            ...prev,
            statusText: `GCD Strategy ${event.strategy}: No Match`,
            statusColor: colors.dimmed,
          }))
          addLog("GCD", `Strategy ${event.strategy}: no match`, colors.dimmed)
          updateLiveStatus("No GCD Matches Found", colors.dimmed)
          break

        case "ComicVineSearching":
          setPipeline(prev => {
            const visited = [...prev.visited]
            visited[STEP_IDX.comicvine] = true
            return {
              ...prev,
              activeStep: STEP_IDX.comicvine,
              visited: visited as boolean[],
              statusText: `Searching ComicVine (${event.stage}) for ${formatSearchFields(searchFieldsRef.current)}`,
              statusColor: colors.magenta,
            }
          })
          updateLiveStatus(`Searching ComicVine (${event.stage}) for ${formatSearchFields(searchFieldsRef.current)}`, colors.magenta)
          addLog("COMICVINE", `${event.stage}...`, colors.magenta)
          break

        case "VisualComparing":
          setPipeline(prev => {
            const visited = [...prev.visited]
            visited[STEP_IDX.compare] = true
            return {
              ...prev,
              activeStep: STEP_IDX.compare,
              visited: visited as boolean[],
              statusText: `Comparing Covers: ${event.title} #${event.issue}`,
              statusColor: colors.magenta,
            }
          })
          updateLiveStatus("Comparing Covers", colors.magenta)
          addLog("COMICVINE", `Comparing covers: ${event.title} #${event.issue}`, colors.magenta)
          break

        case "ComicVineMatchFound":
          matchSourceForImageRef.current = "ComicVine"
          setPipeline(prev => ({
            ...prev,
            statusText: `ComicVine Candidate: ${event.title} (Score: ${event.confidence.toFixed(1)})`,
            statusColor: colors.magenta,
          }))
          addLog("COMICVINE", `Match: ${event.title} (${event.confidence.toFixed(1)})`, colors.magenta)
          updateLiveStatus(`ComicVine Candidate Found: ${event.title}`, colors.magenta)
          break

        case "ScanComplete": {
          setDone(d => d + 1)
          const r = event.result
          const confidence = r.confidence || event.confidence
          const boxMax = previewBoxesRef.current.find((box) => box.name === r.box)?.count
          const nextProcessed = (processedByBoxRef.current[r.box] ?? 0) + 1
          processedByBoxRef.current[r.box] = boxMax ? Math.min(nextProcessed, boxMax) : nextProcessed
          setGlobalProcessed((count) => {
            const total = globalTotalRef.current
            return total > 0 ? Math.min(count + 1, total) : count + 1
          })
          persistMetadataFast()
          const resultFields: SearchFields = {
            title: r.title,
            issue: r.issue_number,
            month: r.month,
            year: r.year,
            publisher: r.publisher,
          }
          searchFieldsRef.current = resultFields
          const resultKey = `${r.box}::${r.filename}`
          const resultImagePath = imagePathByResultKeyRef.current.get(resultKey) ?? currentImagePathRef.current

          setRunResults(prev => {
            const row: RunResultRow = {
              key: resultKey,
              title: r.title,
              issue_number: r.issue_number,
              month: r.month,
              year: r.year,
              publisher: r.publisher,
              box: r.box,
              filename: r.filename,
              confidence,
              imagePath: resultImagePath ?? null,
            }

            const existingIndex = prev.findIndex(item => item.key === resultKey)
            if (existingIndex >= 0) {
              const next = [...prev]
              next[existingIndex] = row
              if (previewModeRef.current === "live") {
                setSelectedResultIndex(existingIndex)
              }
              return next
            }

            const next = [...prev, row]
            if (previewModeRef.current === "live") {
              setSelectedResultIndex(next.length - 1)
            }
            return next
          })

          addLog("RESULT", `${r.title} #${r.issue_number} (${r.year}) — ${confidence}`, confidenceColor(confidence))
          const resultText = formatSearchFields(resultFields)

          // Distinguish "no match" (only VLM data, never searched) from "low confidence match"
          const searched = pipelineRef.current.visited[STEP_IDX.gcd] || pipelineRef.current.visited[STEP_IDX.comicvine]
          let resultStatusText: string
          let resultStatusColor: string
          let resultConfidence = confidence
          if (confidence === "high") {
            resultStatusText = `High Confidence Match Found: ${resultText}`
            resultStatusColor = colors.green
          } else if (confidence === "medium") {
            resultStatusText = `Medium Confidence Match Found: ${resultText}`
            resultStatusColor = colors.yellow
          } else if (searched) {
            resultStatusText = `Low Confidence Match Found: ${resultText}`
            resultStatusColor = colors.red
          } else {
            resultStatusText = `No Match Found - Using VLM Data: ${resultText}`
            resultStatusColor = colors.red
            resultConfidence = "none"
          }
          const isMatch = confidence === "high" || confidence === "medium" || (confidence === "low" && searched)
          if (isMatch) {
            const source = matchSourceForImageRef.current
              ?? (pipelineRef.current.visited[STEP_IDX.comicvine] ? "ComicVine" : pipelineRef.current.visited[STEP_IDX.gcd] ? "GCD" : "Search")
            const now = Date.now()
            setLatestMatch({
              title: r.title,
              issue: r.issue_number,
              month: r.month,
              year: r.year,
              publisher: r.publisher,
              confidence,
              source,
              foundAtMs: now,
            })
          }
          setPipeline(prev => ({
            ...prev,
            activeStep: -1,
            confidence: resultConfidence,
            statusText: resultStatusText,
            statusColor: resultStatusColor,
          }))
          updateLiveStatus(resultStatusText, resultStatusColor)
          if (isConfidenceLevel(confidence)) {
            setStats(s => ({ ...s, [confidence]: s[confidence] + 1 }))
          }
          break
        }

        case "ScanError":
          addLog("ERROR", `${event.filename}: ${event.error}`, colors.red)
          updateLiveStatus(`Error Processing ${event.filename}: ${event.error}`, colors.red)
          setPipeline(prev => ({
            ...prev,
            activeStep: -1,
            statusText: `Error: ${event.error}`,
            statusColor: colors.red,
            confidence: "",
          }))
          setDone(d => d + 1)
          break

        case "BoxFinished":
          addLog("BOX", `Finished ${event.box_name}`, colors.accent)
          updateLiveStatus(`Finished ${event.box_name}`, colors.accent)
          persistMetadataFast()
          break

        case "paused":
          setPhase("paused")
          addLog("PAUSED", "Run stopped at checkpoint — [Enter] Resume from TSV", colors.yellow)
          updateLiveStatus("Run Paused. [Enter] Resume.", colors.yellow)
          persistMetadataFast("paused")
          break

        case "resumed":
          setPhase("running")
          addLog("RESUMED", "Run resumed", colors.green)
          updateLiveStatus("Run Resumed", colors.green)
          persistMetadataFast("running")
          break

        case "run_complete":
          setPhase("complete")
          addLog("COMPLETE", "All boxes finished — run summary ready", colors.green)
          updateLiveStatus("Run Complete. Review summary and press [Enter] to return to menu.", colors.green)
          void refreshMetadataIdentity("complete")
          break

        case "cancelled":
          updateLiveStatus("Run Cancelled.", colors.yellow)
          void refreshMetadataIdentity("cancelled")
          onQuit()
          break

        case "error":
          addLog("ERROR", `IPC error (${event.command}): ${event.message}`, colors.red)
          updateLiveStatus(`IPC Error (${event.command}): ${event.message}`, colors.red)
          break
      }
    })
  }, [onEvent, addLog, autoStart, sendCommand, updateLiveStatus, resume, persistMetadataFast, refreshMetadataIdentity])

  // Keyboard controls — single manually-managed listener via renderer Core API.
  useEffect(() => {
    const handler = (key: any) => {
      // Skip release events (raw API includes them unlike useKeyboard)
      if (key.eventType === "release") return

      const currentPhase = phaseRef.current

      if (key.name === "b" && (currentPhase === "running" || currentPhase === "paused")) {
        if (previewModeRef.current === "live") {
          setPreviewMode("browse")
          if (runResultsRef.current.length > 0) {
            setSelectedResultIndex(i => i >= 0 ? i : runResultsRef.current.length - 1)
          }
          addLog("SYSTEM", "Browse mode enabled. [↑/↓] Navigate results.", colors.yellow)
          updateLiveStatus("Browse Mode Enabled", colors.yellow)
        } else {
          setPreviewMode("live")
          addLog("SYSTEM", "Live preview mode enabled", colors.green)
          updateLiveStatus("Live Preview Mode Enabled", colors.green)
        }
      }

      if (previewModeRef.current === "browse" && runResultsRef.current.length > 0) {
        if (key.name === "up") {
          setSelectedResultIndex(i => Math.max(0, i - 1))
          return
        }
        if (key.name === "down") {
          setSelectedResultIndex(i => Math.min(runResultsRef.current.length - 1, i + 1))
          return
        }
        if (key.name === "pageup") {
          setSelectedResultIndex(i => Math.max(0, i - 8))
          return
        }
        if (key.name === "pagedown") {
          setSelectedResultIndex(i => Math.min(runResultsRef.current.length - 1, i + 8))
          return
        }
        if (key.name === "home") {
          setSelectedResultIndex(0)
          return
        }
        if (key.name === "end") {
          setSelectedResultIndex(runResultsRef.current.length - 1)
          return
        }
      }

      if (key.name === "return" || key.name === "enter") {
        if (currentPhase === "ready") {
          if (resume && !resumeRowsReadyRef.current) {
            addLog("SYSTEM", "Resume checkpoint is still loading. Please wait...", colors.yellow)
            updateLiveStatus("Resume Checkpoint Is Still Loading. Please Wait...", colors.yellow)
            return
          }
          if (!metadataReadyRef.current) {
            addLog("SYSTEM", "Run metadata is still initializing. Please wait...", colors.yellow)
            updateLiveStatus("Run Metadata Is Still Initializing. Please Wait...", colors.yellow)
            return
          }
          sendCommand({ cmd: "scan", resume })
          setPhase("running")
          setPreviewMode("live")
          if (!resume) {
            setRunResults([])
            setSelectedResultIndex(-1)
            setStats({ high: 0, medium: 0, low: 0 })
          }
          imagePathByResultKeyRef.current.clear()
          addLog("SYSTEM", resume ? "Resume scan started" : "Scan started", colors.green)
        } else if (currentPhase === "paused") {
          sendCommand({ cmd: "scan", resume: true })
          setPhase("running")
          addLog("RESUMED", "Resuming from TSV checkpoint", colors.green)
        } else if (currentPhase === "complete") {
          onComplete()
        }
      }

      if (key.name === "p" && currentPhase === "running") {
        // Pause behaves like checkpoint-stop: terminate current run and resume later.
        sendCommand({ cmd: "pause" })
        setPhase("paused")
        addLog("PAUSED", "Paused. [Enter] Resume from checkpoint. [Esc] Cancel (double press).", colors.yellow)
      }

      if (key.name === "escape") {
        if (currentPhase === "complete") {
          onComplete()
          return
        }
        if (escPressedRef.current) {
          if (escTimerRef.current) clearTimeout(escTimerRef.current)
          escPressedRef.current = false
          setEscPromptActive(false)
          // Optimistic: navigate away immediately, backend processes cancel async
          sendCommand({ cmd: "cancel" })
          onQuit()
        } else {
          escPressedRef.current = true
          setEscPromptActive(true)
          addLog("SYSTEM", "Press [Esc] again to cancel", colors.yellow)
          escTimerRef.current = setTimeout(() => {
            escPressedRef.current = false
            setEscPromptActive(false)
          }, 2000)
        }
      }
    }

    renderer.keyInput.on("keypress", handler)
    return () => { renderer.keyInput.off("keypress", handler) }
  }, [sendCommand, onComplete, onQuit, addLog, updateLiveStatus, resume])

  useEffect(() => {
    return () => {
      if (escTimerRef.current) clearTimeout(escTimerRef.current)
    }
  }, [])

  const overallProgress = globalTotal > 0 ? Math.round((globalProcessed / globalTotal) * 100) : 0
  const boxProgress = boxTotal > 0 ? Math.round((done / boxTotal) * 100) : 0
  const cancelEscHint = escPromptActive ? "[Esc] Confirm Cancel" : "[Esc] Cancel"

  // Pre-scan phase
  if (phase === "pre_scan") {
    return (
      <box flexDirection="column" width="100%" height="100%">
        <box flexGrow={1} justifyContent="center" alignItems="center">
          <text fg={colors.dimmed}>
            {resume && !resumeRowsReady
              ? "Loading resume checkpoint..."
              : !metadataReady
                ? "Preparing run metadata..."
                : "Scanning directories..."}
          </text>
        </box>
        <StatusBar left="Counting Images..." right={cancelEscHint} />
      </box>
    )
  }

  // Ready phase
  if (phase === "ready") {
    const boxCountText = previewBoxes.length === 1
      ? `in ${previewBoxes[0].name}`
      : `across ${previewBoxes.length} boxes`
    const readyActionText = resume ? "[Enter] Resume Run" : "[Enter] Start Run"
    const readyTitle = resume
      ? `${totalImages} images found ${boxCountText} (resume mode)`
      : `${totalImages} images found ${boxCountText}`

    return (
      <box flexDirection="column" width="100%" height="100%">
        <box flexGrow={1} justifyContent="center" alignItems="center" flexDirection="column" gap={2}>
          <text fg={colors.white}>
            <strong>{readyTitle}</strong>
          </text>
          <text fg={colors.accent}>{readyActionText}</text>
        </box>
        <StatusBar
          left={readyActionText}
          right={cancelEscHint}
        />
      </box>
    )
  }

  if (phase === "complete") {
    const imagesInScope = globalTotal || totalImages
    const scopeCount = previewBoxes.length > 0 ? previewBoxes.length : totalBoxes
    const runModeLabel = runMode === "single_box" ? "Single Box" : "Batch"
    const selectedFolder = singleBoxDir || currentBox || "(not set)"
    const resultsPath = outputTsvPath || "(not set)"
    const totalProcessed = stats.high + stats.medium + stats.low
    const showTextLogo = textLogoRaw && height >= 18
    const showEyeLogo = eyeLogoRows && width >= 85 && height >= 42

    return (
      <box flexDirection="column" width="100%" height="100%">
        <box flexDirection="column" alignItems="center" flexGrow={1}>
          {showTextLogo && (
            <box flexDirection="column" alignItems="center" paddingTop={1}>
              {showEyeLogo && <AnsiArt rows={eyeLogoRows} />}
              <text fg={colors.fg}>{textLogoRaw}</text>
            </box>
          )}
          <box flexGrow={1} />
          <text fg={colors.white}>Run Complete Summary</text>
          <box flexDirection="column" marginTop={1}>
            <text fg={colors.fg}>{"  TSV Output:         "}<span fg={colors.accent}>{resultsPath}</span></text>
            <text fg={colors.fg}>{"  Input Directory:    "}<span fg={colors.accent}>{inputRootDir || "(not set)"}</span></text>
            <text fg={colors.fg}>{"  Run Mode:           "}<span fg={colors.accent}>{runModeLabel}</span></text>
            {runMode === "batch"
              ? <text fg={colors.fg}>{"  Folders in Scope:   "}<span fg={colors.accent}>{String(scopeCount)}</span></text>
              : <text fg={colors.fg}>{"  Selected Folder:    "}<span fg={colors.accent}>{selectedFolder}</span></text>}
            <text fg={colors.fg}>{"  Images Processed:   "}<span fg={colors.accent}>{`${globalProcessed}/${imagesInScope || globalProcessed}`}</span></text>
            <text fg={colors.fg}>{"  High Confidence:    "}<span fg={colors.green}>{`${stats.high}  ${totalProcessed > 0 ? Math.round((stats.high / totalProcessed) * 100) : 0}%`}</span></text>
            <text fg={colors.fg}>{"  Medium Confidence:  "}<span fg={colors.yellow}>{`${stats.medium}  ${totalProcessed > 0 ? Math.round((stats.medium / totalProcessed) * 100) : 0}%`}</span></text>
            <text fg={colors.fg}>{"  Low Confidence:     "}<span fg={colors.red}>{`${stats.low}  ${totalProcessed > 0 ? Math.round((stats.low / totalProcessed) * 100) : 0}%`}</span></text>
          </box>
          <text fg={colors.dimmed} marginTop={2}>Review medium and low confidence matches for accuracy.</text>
          <text fg={colors.accent} marginTop={2}>[Enter] Back to Menu</text>
          <box flexGrow={1} />
        </box>
        <StatusBar left="[Enter] Back to Menu" right="[Esc] Back to Menu" />
      </box>
    )
  }

  // Determine spinner color for active pipeline step
  const spinnerActive = pipeline.activeStep >= 0
  const spinnerColor = spinnerActive ? STEP_COLORS[pipeline.activeStep] : colors.dimmed

  // Running / Paused phase
  const statusLeft = phase === "paused" ? "[Enter] Resume" : "[P] Pause"
  const browseHint = previewMode === "browse"
    ? "[B] Live Preview  [↑/↓] Scroll"
    : "[B] Browse Results"
  const statusRight = escPromptActive ? "[Esc] Confirm Cancel" : "[Esc] Cancel"

  const selectedBrowseRow = previewMode === "browse" && selectedResultIndex >= 0
    ? runResults[selectedResultIndex] ?? null
    : null
  const previewImagePath = previewMode === "browse"
    ? selectedBrowseRow?.imagePath ?? null
    : currentImagePath

  const statusBarLeft = `${statusLeft}  ⋄  ${browseHint}`
  const statusBarHeight = statusBarRows(statusBarLeft, statusRight, width)
  const isBatchHeader = runMode === "batch" && totalBoxes > 1
  const headerHeight = isBatchHeader ? 6 : 5
  const latestMatchViewportHeight = 7
  const splitMarginTop = 1
  const latestMatchMarginTop = 1
  const availableHeight = Math.max(
    6,
    height - headerHeight - splitMarginTop - latestMatchMarginTop - latestMatchViewportHeight - statusBarHeight,
  )
  const splitContentWidth = Math.max(28, width - 2)
  const leftPaneWidth = Math.max(18, Math.min(Math.floor(splitContentWidth * 0.47), splitContentWidth - 21))
  const rightPaneWidth = Math.max(20, splitContentWidth - leftPaneWidth - 1)
  const rightPaneInnerWidth = Math.max(12, rightPaneWidth - 4)
  const previewWidth = Math.max(16, leftPaneWidth - 4)
  const previewHeight = Math.max(4, availableHeight - 2)

  const resultsListHeight = Math.max(2, availableHeight - 5)
  const safeSelectedIndex = runResults.length === 0
    ? -1
    : Math.min(Math.max(selectedResultIndex, 0), runResults.length - 1)
  const startIndex = safeSelectedIndex < 0
    ? 0
    : Math.max(0, Math.min(safeSelectedIndex - Math.floor(resultsListHeight / 2), runResults.length - resultsListHeight))
  const visibleRows = runResults.slice(startIndex, startIndex + resultsListHeight)

  const issueW = 6
  const monthW = 5
  const yearW = 6
  const publisherW = 12
  const confidenceW = 8
  const fixedColumnsWidth = issueW + monthW + yearW + publisherW + confidenceW + 7
  const titleW = Math.max(8, rightPaneInnerWidth - fixedColumnsWidth)

  const latestMatchContentWidth = Math.max(12, width - 6)
  const headerStatusText = formatHeaderStatus(pipeline.statusText || liveStatus.text)
  const selectedCountText = runResults.length === 0 ? "0/0" : `${safeSelectedIndex + 1}/${runResults.length}`
  const resultsFooter = buildResultsFooterLayout(selectedCountText, stats, rightPaneInnerWidth)

  return (
    <box flexDirection="column" width="100%" height="100%">
      {/* TOP: Bordered status header + progress + pipeline steps */}
      {(() => {
        const isBatch = runMode === "batch" && totalBoxes > 1
        const boxLabel = isBatch
          ? `${currentBox || "Box_??"}/${totalBoxes}`
          : (currentBox || "Box_??")
        const totalLabel = "Total"
        const labelWidth = isBatch
          ? Math.max(boxLabel.length, totalLabel.length)
          : boxLabel.length
        const paddedBoxLabel = boxLabel.padEnd(labelWidth)
        const paddedTotalLabel = totalLabel.padEnd(labelWidth)
        const boxCount = `${done}/${boxTotal}`
        const totalCount = `${globalProcessed}/${globalTotal || totalImages}`
        const countWidth = isBatch
          ? Math.max(boxCount.length, totalCount.length)
          : boxCount.length
        const paddedBoxCount = boxCount.padStart(countWidth)
        const paddedTotalCount = totalCount.padStart(countWidth)
        // Bar width: total width - margins(2) - borders(2) - paddingX(2) - label - " : " - count - " " - " " - pct(4)
        const barWidth = Math.max(10, width - 6 - labelWidth - 3 - countWidth - 2 - 5)

        return (
          <box border borderStyle="single" title={isBatch ? " Batch Run " : " Single Box Run "} flexDirection="column" marginX={1} marginTop={0}>
            {/* Row 1: Spinner + contextual status */}
            <box flexDirection="row" paddingX={1} gap={1} alignItems="center">
              {spinnerActive ? (
                <RuneSpinner color={spinnerColor} playing={phase !== "paused"} />
              ) : (
                <box width={1} height={1} />
              )}
              <text fg={pipeline.statusText ? pipeline.statusColor : liveStatus.color}>
                {clip(headerStatusText, Math.max(10, width - 11))}
              </text>
            </box>

            {/* Separator */}
            <box height={1} />

            {/* Box progress row */}
            <box flexDirection="row" paddingX={1} gap={1} alignItems="center">
              <text fg={colors.dimmed}>{paddedBoxLabel}{" : "}{paddedBoxCount}</text>
              <BrailleProgressBar
                progress={boxProgress}
                width={barWidth}
                playing={phase === "running"}
              />
            </box>

            {/* Total progress row (batch only) */}
            {isBatch && (
              <box flexDirection="row" paddingX={1} gap={1} alignItems="center">
                <text fg={colors.dimmed}>{paddedTotalLabel}{" : "}{paddedTotalCount}</text>
                <BrailleProgressBar
                  progress={overallProgress}
                  width={barWidth}
                  playing={phase === "running"}
                />
              </box>
            )}
          </box>
        )
      })()}

      {/* BOTTOM: Split left (preview) / right (scrollable TSV results) */}
      <box flexDirection="row" height={availableHeight} paddingX={1} marginTop={splitMarginTop} gap={1}>
        {/* Left: ANSI image preview */}
        <box
          width={leftPaneWidth}
          height={availableHeight}
          border
          borderStyle="single"
          title=" Image Preview "
          paddingX={1}
        >
          <ImagePreview
            imagePath={previewImagePath}
            availableWidth={previewWidth}
            availableHeight={previewHeight}
          />
        </box>

        {/* Right: scrollable TSV-style results browser */}
        <box
          width={rightPaneWidth}
          height={availableHeight}
          border
          borderStyle="single"
          title={` Results (${runResults.length}) `}
          flexDirection="column"
          paddingX={1}
        >
          <text fg={colors.dimmed}>
            {" ".repeat(2)}
            {clip("Title", titleW)} {clip("Issue", issueW)} {clip("Month", monthW)} {clip("Year", yearW)} {clip("Publisher", publisherW)} {clip("Conf", confidenceW)}
          </text>

          <box flexDirection="column" flexGrow={1}>
            {visibleRows.length === 0 ? (
              <text fg={colors.dimmed}>Waiting for scan results...</text>
            ) : (
              visibleRows.map((row, i) => {
                const actualIndex = startIndex + i
                const selected = previewMode === "browse" && actualIndex === safeSelectedIndex
                const prefix = selected ? ">" : " "
                const rowBg = selected ? colors.accent : confidenceBackground(row.confidence)
                const line = `${prefix} ${clip(row.title, titleW)} ${clip(row.issue_number, issueW)} ${clip(row.month, monthW)} ${clip(row.year, yearW)} ${clip(row.publisher, publisherW)} ${clip(row.confidence, confidenceW)}`
                return (
                  <box key={row.key} width="100%" height={1}>
                    <text fg={selected ? colors.white : colors.fg} bg={rowBg} width="100%">{line}</text>
                  </box>
                )
              })
            )}
          </box>

          <box flexDirection="row" height={1} backgroundColor={colors.bg}>
            <text fg={colors.dimmed}>{resultsFooter.count}</text>
            <text fg={colors.dimmed}>{resultsFooter.gap}</text>
            {resultsFooter.high && <text fg={colors.green}>{resultsFooter.high}</text>}
            {resultsFooter.medium && (
              <>
                <text fg={colors.dimmed}>{resultsFooter.separator}</text>
                <text fg={colors.yellow}>{resultsFooter.medium}</text>
              </>
            )}
            {resultsFooter.low && (
              <>
                <text fg={colors.dimmed}>{resultsFooter.separator}</text>
                <text fg={colors.red}>{resultsFooter.low}</text>
              </>
            )}
          </box>
        </box>
      </box>

      <box
        border
        borderStyle="single"
        title={previewMode === "browse" ? " Browse " : " Latest Match "}
        flexDirection="column"
        marginX={1}
        marginTop={latestMatchMarginTop}
        height={latestMatchViewportHeight}
        paddingX={1}
        justifyContent="center"
      >
        <RuneRevealMatch
          match={latestMatch}
          width={latestMatchContentWidth}
          mode={previewMode}
          browseData={selectedBrowseRow ? {
            title: selectedBrowseRow.title,
            issue: selectedBrowseRow.issue_number,
            month: selectedBrowseRow.month,
            year: selectedBrowseRow.year,
            publisher: selectedBrowseRow.publisher,
            confidence: selectedBrowseRow.confidence,
            filename: selectedBrowseRow.filename,
            imagePath: selectedBrowseRow.imagePath,
          } : null}
          selectedIndex={safeSelectedIndex}
        />
      </box>

      <StatusBar left={statusBarLeft} right={statusRight} />
    </box>
  )
}
