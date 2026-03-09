import { useState, useEffect, useRef } from "react"
import { useRenderer, useTerminalDimensions } from "@opentui/react"
import { homedir } from "os"
import { DirectoryPicker } from "./DirectoryPicker"
import { AnsiArt } from "./AnsiArt"
import { StatusBar } from "./StatusBar"
import { colors } from "../theme"
import type { Config, Command, ScanEvent, BoxInfo, RunMode, ResumeStats } from "../types"
import type { AnsiSegment } from "../utils/ansiArt"
import { analyzeResumeTsv } from "../utils/resumeRun"

interface NewRunWizardProps {
  mode: "new" | "resume"
  eyeLogoRows: AnsiSegment[][]
  textLogoRaw: string
  config: Config
  sendCommand: (cmd: Command) => void
  onEvent: (listener: (event: ScanEvent) => void) => () => void
  onStartRun: (context: {
    inputRootDir: string
    outputTsvPath: string
    runMode: RunMode
    singleBoxDir: string
    resumeStats: ResumeStats | null
  }) => void
  onCancel: () => void
}

type WizardStep = "dir_select" | "mode" | "box_select" | "output_path" | "resume_path" | "confirm"

export function NewRunWizard({ mode, eyeLogoRows, textLogoRaw, config, sendCommand, onEvent, onStartRun, onCancel }: NewRunWizardProps) {
  const renderer = useRenderer()
  const { width, height } = useTerminalDimensions()
  const isResumeWizard = mode === "resume"
  const pickerStartPath = homedir()
  const eyeLogoWidth = Math.max(...eyeLogoRows.map(row => row.reduce((sum, segment) => sum + segment.text.length, 0)))
  const eyeLogoHeight = eyeLogoRows.length
  const textLogoHeight = textLogoRaw.split("\n").length

  const [step, setStep] = useState<WizardStep>(isResumeWizard ? "resume_path" : "dir_select")
  const [inputRootDir, setInputRootDir] = useState(isResumeWizard ? config.input_root_dir : "")
  const [runMode, setRunMode] = useState<RunMode>(config.run_mode === "single_box" ? "single_box" : "batch")
  const [modeIndex, setModeIndex] = useState(0)
  const [singleBoxDir, setSingleBoxDir] = useState(isResumeWizard ? config.single_box_dir : "")
  const [outputTsvPath, setOutputTsvPath] = useState(isResumeWizard ? config.output_tsv_path : "")
  const [candidateFolders, setCandidateFolders] = useState<string[]>([])
  const [folderCursor, setFolderCursor] = useState(0)
  const [previewData, setPreviewData] = useState<{ total_images: number; boxes: BoxInfo[] } | null>(null)
  const [resumeStats, setResumeStats] = useState<ResumeStats | null>(null)
  const [pathError, setPathError] = useState<string | null>(null)

  const stepRef = useRef(step)
  stepRef.current = step

  function getDefaultOutputPath(dir: string, mode: RunMode, boxDir: string): string {
    if (mode === "single_box" && boxDir) {
      const boxName = boxDir.split("/").pop() || "Box"
      return `${boxDir}/${boxName}.tsv`
    }
    return `${dir}/All_Boxes.tsv`
  }

  function persistWizardConfigAndPreview(overrides?: {
    inputRootDir?: string
    runMode?: RunMode
    singleBoxDir?: string
    outputTsvPath?: string
  }) {
    const resolvedInputRootDir = (overrides?.inputRootDir ?? inputRootDir).trim()
    const resolvedRunMode = overrides?.runMode ?? runMode
    const resolvedSingleBoxDir = (overrides?.singleBoxDir ?? singleBoxDir).trim()
    const resolvedOutputPath = (overrides?.outputTsvPath ?? outputTsvPath).trim()
      || getDefaultOutputPath(resolvedInputRootDir, resolvedRunMode, resolvedSingleBoxDir)
    const selectedBoxDir = resolvedRunMode === "single_box" ? resolvedSingleBoxDir : ""

    sendCommand({
      cmd: "set-config",
      config: {
        input_root_dir: resolvedInputRootDir,
        run_mode: resolvedRunMode,
        single_box_dir: selectedBoxDir,
        output_tsv_path: resolvedOutputPath,
      },
    })
    sendCommand({ cmd: "scan-preview" })
  }

  function confirmOutputPath(pathValue?: string) {
    setPathError(null)
    const resolvedPath = (pathValue ?? outputTsvPath).trim() || getDefaultOutputPath(inputRootDir, runMode, singleBoxDir)
    setOutputTsvPath(resolvedPath)
    setPreviewData(null)
    persistWizardConfigAndPreview({ outputTsvPath: resolvedPath })
    setStep("confirm")
  }

  async function confirmResumePath(pathValue?: string) {
    setPathError(null)
    setPreviewData(null)
    setResumeStats(null)

    try {
      const analysis = await analyzeResumeTsv(pathValue ?? outputTsvPath, config)
      setInputRootDir(analysis.inputRootDir)
      setRunMode(analysis.runMode)
      setSingleBoxDir(analysis.singleBoxDir)
      setOutputTsvPath(analysis.outputTsvPath)
      setResumeStats({
        runId: analysis.runId,
        processedImages: analysis.processedImages,
        processedBoxCount: analysis.processedBoxCount,
        processedByBox: analysis.processedByBox,
        uniqueBoxes: analysis.uniqueBoxes,
        completedBoxes: analysis.completedBoxes,
        metadataMatchedBy: analysis.metadataMatchedBy,
      })
      persistWizardConfigAndPreview({
        inputRootDir: analysis.inputRootDir,
        runMode: analysis.runMode,
        singleBoxDir: analysis.singleBoxDir,
        outputTsvPath: analysis.outputTsvPath,
      })
      setStep("confirm")
    } catch (error: any) {
      const message = error?.message ?? String(error)
      setPathError(message)
    }
  }

  useEffect(() => {
    return onEvent((event) => {
      if (event.event === "scan_preview") {
        setPreviewData({ total_images: event.total_images, boxes: event.boxes })
      }
      if (event.event === "dirs" && stepRef.current === "box_select") {
        const dirEvent = event as { event: "dirs"; path: string; dirs: string[] }
        setCandidateFolders(dirEvent.dirs)
        setFolderCursor(0)
      }
    })
  }, [onEvent])

  useEffect(() => {
    if (step !== "mode" && step !== "box_select" && step !== "output_path" && step !== "resume_path" && step !== "confirm") return

    const handler = (key: any) => {
      if (key.eventType === "release") return

      if (key.name === "escape") {
        if (step === "resume_path") onCancel()
        else if (step === "mode") setStep("dir_select")
        else if (step === "box_select") setStep("mode")
        else if (step === "output_path") {
          if (runMode === "single_box") setStep("box_select")
          else setStep("mode")
        } else if (step === "confirm") {
          setStep(isResumeWizard ? "resume_path" : "output_path")
        }
        return
      }

      if (key.name === "q" && (step === "mode" || step === "box_select" || step === "resume_path" || step === "confirm")) {
        onCancel()
        return
      }

      if (step === "mode") {
        if (key.name === "up" || key.name === "down") {
          setModeIndex(i => i === 0 ? 1 : 0)
          return
        }
        if (key.name === "return" || key.name === "enter") {
          const selected: RunMode = modeIndex === 0 ? "single_box" : "batch"
          setRunMode(selected)
          setPreviewData(null)

          if (selected === "single_box") {
            sendCommand({ cmd: "list-dirs", path: inputRootDir })
            setStep("box_select")
          } else {
            setSingleBoxDir("")
            setOutputTsvPath(getDefaultOutputPath(inputRootDir, "batch", ""))
            setStep("output_path")
          }
        }
        return
      }

      if (step === "box_select") {
        if (key.name === "up") {
          setFolderCursor(i => Math.max(0, i - 1))
          return
        }
        if (key.name === "down") {
          setFolderCursor(i => Math.min(candidateFolders.length - 1, i + 1))
          return
        }
        if (key.name === "return" || key.name === "enter") {
          if (candidateFolders.length > 0) {
            const selected = candidateFolders[folderCursor]
            const fullPath = `${inputRootDir}/${selected}`
            setSingleBoxDir(fullPath)
            setOutputTsvPath(getDefaultOutputPath(inputRootDir, "single_box", fullPath))
            setPreviewData(null)
            setStep("output_path")
          }
        }
        return
      }

      // Enter key naming can vary across terminals ("enter", "return", "linefeed", "kpenter").
      // Confirm at wizard level so this step cannot hang on key-name differences.
      if (step === "output_path" && (key.name === "enter" || key.name === "return" || key.name === "linefeed" || key.name === "kpenter")) {
        confirmOutputPath()
        return
      }

      if (step === "resume_path" && (key.name === "enter" || key.name === "return" || key.name === "linefeed" || key.name === "kpenter")) {
        void confirmResumePath()
        return
      }

      if (step === "confirm" && (key.name === "return" || key.name === "enter")) {
        onStartRun({
          inputRootDir,
          outputTsvPath,
          runMode,
          singleBoxDir,
          resumeStats,
        })
      }
    }

    renderer.keyInput.on("keypress", handler)
    return () => { renderer.keyInput.off("keypress", handler) }
  }, [step, modeIndex, candidateFolders, folderCursor, inputRootDir, runMode, singleBoxDir, outputTsvPath, onCancel, onStartRun, isResumeWizard, config])

  const showTextLogo = height >= textLogoHeight + 8
  const showEyeLogo = width >= eyeLogoWidth + 2 && height >= eyeLogoHeight + textLogoHeight + 12
  const selectorWidth = Math.max(24, Math.min(width - 4, 110))
  const selectorHeight = Math.max(10, height - (showTextLogo ? (showEyeLogo ? 20 : 10) : 6))
  const outputInputWidth = Math.max(30, Math.min(width - 8, 100))

  const maxBoxVisible = Math.max(6, Math.min(candidateFolders.length || 6, height - (showTextLogo ? (showEyeLogo ? 22 : 12) : 8)))
  const boxHalfWindow = Math.floor(maxBoxVisible / 2)
  let boxStartIdx = Math.max(0, folderCursor - boxHalfWindow)
  const boxEndIdx = Math.min(candidateFolders.length, boxStartIdx + maxBoxVisible)
  if (boxEndIdx - boxStartIdx < maxBoxVisible) boxStartIdx = Math.max(0, boxEndIdx - maxBoxVisible)

  const renderHeader = () => {
    if (!showTextLogo) return null
    return (
      <box flexDirection="column" alignItems="center" paddingTop={1}>
        {showEyeLogo && <AnsiArt rows={eyeLogoRows} />}
        <text fg={colors.fg}>{textLogoRaw}</text>
      </box>
    )
  }

  if (step === "dir_select") {
    return (
      <box flexDirection="column" width="100%" height="100%">
        {renderHeader()}
        <box flexGrow={1} alignItems="center" justifyContent="center" paddingX={2} paddingY={1}>
          <box width={selectorWidth} height={selectorHeight}>
            <DirectoryPicker
              startPath={pickerStartPath}
              sendCommand={sendCommand}
              onEvent={onEvent}
              onSelect={(path) => {
                setInputRootDir(path)
                setStep("mode")
              }}
              onCancel={onCancel}
            />
          </box>
        </box>
      </box>
    )
  }

  if (step === "mode") {
    return (
      <box flexDirection="column" width="100%" height="100%">
        {renderHeader()}
        <box flexDirection="column" alignItems="center" justifyContent="center" flexGrow={1}>
          <text fg={colors.dimmed}>Selected: {inputRootDir}</text>
          <text fg={colors.white} marginTop={1}>Select run mode:</text>
          <box flexDirection="column" marginTop={1}>
            <box flexDirection="row" height={3}>
              <text fg={modeIndex === 0 ? colors.accent : colors.dimmed}>
                {modeIndex === 0 ? " > " : "   "}
              </text>
              <box flexDirection="column">
                <text fg={modeIndex === 0 ? colors.highlight : colors.dimmed}>SINGLE BOX MODE</text>
                <text fg={colors.dimmed}>Process one selected folder</text>
              </box>
            </box>
            <box flexDirection="row" height={3}>
              <text fg={modeIndex === 1 ? colors.accent : colors.dimmed}>
                {modeIndex === 1 ? " > " : "   "}
              </text>
              <box flexDirection="column">
                <text fg={modeIndex === 1 ? colors.highlight : colors.dimmed}>BATCH MODE</text>
                <text fg={colors.dimmed}>Process all folders under root directory</text>
              </box>
            </box>
          </box>
        </box>
        <StatusBar left="[↑/↓] Navigate  ⋄  [Enter] Select" right="[Esc] Back  ⋄  [Q] Cancel" />
      </box>
    )
  }

  if (step === "box_select") {
    return (
      <box flexDirection="column" width="100%" height="100%">
        {renderHeader()}
        <box flexDirection="column" alignItems="center" justifyContent="center" flexGrow={1} gap={1}>
          <text fg={colors.white}>Select box folder</text>
          <text fg={colors.dimmed}>{inputRootDir}</text>
          <box width={selectorWidth} height={maxBoxVisible + 2} border borderStyle="single" paddingX={1} paddingY={0}>
            {candidateFolders.length === 0 ? (
              <text fg={colors.dimmed}>Loading folders...</text>
            ) : (
              candidateFolders.slice(boxStartIdx, boxEndIdx).map((folder, i) => {
                const actualIdx = boxStartIdx + i
                return (
                  <text key={folder} fg={actualIdx === folderCursor ? colors.accent : colors.fg}>
                    {actualIdx === folderCursor ? " > " : "   "}{folder}
                  </text>
                )
              })
            )}
          </box>
        </box>
        <StatusBar left="[↑/↓] Navigate  ⋄  [Enter] Select" right="[Esc] Back  ⋄  [Q] Cancel" />
      </box>
    )
  }

  if (step === "resume_path") {
    return (
      <box flexDirection="column" width="100%" height="100%">
        {renderHeader()}
        <box flexDirection="column" alignItems="center" justifyContent="center" flexGrow={1} gap={1}>
          <text fg={colors.white}>Resume TSV Path</text>
          <text fg={colors.dimmed}>
            {config.output_tsv_path.trim()
              ? "Configured TSV path is pre-filled. Press [Enter] to accept it or edit it to choose a different resume file."
              : "Enter the TSV you want to use for resume mode."}
          </text>
          <box flexDirection="column" marginTop={1} gap={1}>
            {config.output_tsv_path.trim() && (
              <text fg={colors.fg}>Configured TSV: <span fg={colors.accent}>{config.output_tsv_path}</span></text>
            )}
            <input
              value={outputTsvPath}
              onInput={(value) => {
                setOutputTsvPath(value)
                if (pathError) setPathError(null)
              }}
              keyBindings={[
                { name: "s", ctrl: true, action: "submit" },
                { name: "enter", action: "submit" },
                { name: "kpenter", action: "submit" },
              ]}
              onSubmit={(value: any) => {
                void confirmResumePath(typeof value === "string" ? value : undefined)
              }}
              focused
              width={outputInputWidth}
            />
            {pathError && (
              <text fg={colors.red}>{pathError}</text>
            )}
          </box>
        </box>
        <StatusBar left="[Enter]/[Ctrl+S] Confirm TSV" right="[Ctrl+Shift+V] Paste  ⋄  [Esc] Cancel  ⋄  [Q] Cancel" />
      </box>
    )
  }

  if (step === "output_path") {
    return (
      <box flexDirection="column" width="100%" height="100%">
        {renderHeader()}
        <box flexDirection="column" alignItems="center" justifyContent="center" flexGrow={1} gap={1}>
          <text fg={colors.white}>Output TSV Path</text>
          <text fg={colors.dimmed}>Default path is pre-filled. Edit it if you want a custom destination.</text>
          <box flexDirection="column" marginTop={1} gap={1}>
            <text fg={colors.fg}>Run Mode: <span fg={colors.accent}>{runMode === "single_box" ? "Single Box" : "Batch"}</span></text>
            {runMode === "single_box" && (
              <text fg={colors.fg}>Selected Box: <span fg={colors.accent}>{singleBoxDir.split("/").pop() || "(none)"}</span></text>
            )}
            <input
              value={outputTsvPath}
              onInput={setOutputTsvPath}
              keyBindings={[
                { name: "s", ctrl: true, action: "submit" },
                { name: "enter", action: "submit" },
                { name: "kpenter", action: "submit" },
              ]}
              onSubmit={(value: any) => confirmOutputPath(typeof value === "string" ? value : undefined)}
              focused
              width={outputInputWidth}
            />
          </box>
        </box>
        <StatusBar left="[Enter]/[Ctrl+S] Confirm Path" right="[Ctrl+Shift+V] Paste  ⋄  [Esc] Back" />
      </box>
    )
  }

  const boxLabel = runMode === "single_box"
    ? singleBoxDir.split("/").pop()
    : `${previewData?.boxes.length ?? "?"} boxes`
  const completedBoxes = resumeStats && previewData
    ? previewData.boxes.filter((box) => (resumeStats.processedByBox[box.name] ?? 0) >= box.count).length
    : 0
  const confirmTitle = isResumeWizard ? "Resume Run Summary" : "Pre-Run Summary"
  const confirmActionText = isResumeWizard ? "[Enter] Resume Run" : "[Enter] Start Run"
  const confirmStatusRight = isResumeWizard ? "[Esc] Back  ⋄  [Q] Cancel" : "[Esc] Back  ⋄  [Q] Cancel"

  return (
    <box flexDirection="column" width="100%" height="100%">
      {renderHeader()}
      <box flexDirection="column" alignItems="center" justifyContent="center" flexGrow={1} gap={1}>
        <text fg={colors.white}>{confirmTitle}</text>
        <box flexDirection="column" marginTop={1}>
          {isResumeWizard && (
            <text fg={colors.fg}>  Resume TSV:       <span fg={colors.accent}>{outputTsvPath}</span></text>
          )}
          <text fg={colors.fg}>  Input Directory:  <span fg={colors.accent}>{inputRootDir}</span></text>
          <text fg={colors.fg}>  Run Mode:         <span fg={colors.accent}>{runMode === "single_box" ? "Single Box" : "Batch"}</span></text>
          {runMode === "single_box" && (
            <text fg={colors.fg}>  Selected Box:     <span fg={colors.accent}>{boxLabel}</span></text>
          )}
          {!isResumeWizard && (
            <text fg={colors.fg}>  Output TSV:       <span fg={colors.accent}>{outputTsvPath}</span></text>
          )}
          {isResumeWizard && resumeStats && runMode === "single_box" && (
            <text fg={colors.fg}>
              {"  Resume Progress:  "}
              <span fg={colors.accent}>
                {resumeStats.processedImages}/{previewData?.total_images ?? "counting..."}
              </span>
              {" images already processed"}
            </text>
          )}
          {isResumeWizard && resumeStats && runMode === "batch" && (
            <text fg={colors.fg}>
              {"  Resume Progress:  "}
              <span fg={colors.accent}>
                {previewData ? completedBoxes : resumeStats.processedBoxCount}/{previewData?.boxes.length ?? "counting..."}
              </span>
              {" boxes complete"}
            </text>
          )}
          {isResumeWizard && resumeStats && runMode === "batch" && (
            <text fg={colors.fg}>
              {"  Images Processed: "}
              <span fg={colors.accent}>
                {resumeStats.processedImages}/{previewData?.total_images ?? "counting..."}
              </span>
            </text>
          )}
          {!isResumeWizard && (
            <text fg={colors.fg}>  Images Found:     <span fg={colors.accent}>{previewData?.total_images ?? "counting..."}</span></text>
          )}
          {isResumeWizard && (
            <text fg={colors.fg}>  Images in Scope:  <span fg={colors.accent}>{previewData?.total_images ?? "counting..."}</span></text>
          )}
          <text fg={colors.fg}>  GCD:              <span fg={config.gcd_enabled ? colors.green : colors.dimmed}>{config.gcd_enabled ? "enabled" : "disabled"}</span></text>
          <text fg={colors.fg}>  ComicVine:        <span fg={config.comicvine_enabled ? colors.green : colors.dimmed}>{config.comicvine_enabled ? "enabled" : "disabled"}</span></text>
          <text fg={colors.fg}>  VLM Endpoint:     <span fg={colors.accent}>{config.vlm_base_url}</span></text>
        </box>
        <text fg={colors.accent} marginTop={2}>{confirmActionText}</text>
      </box>
      <StatusBar left={confirmActionText} right={confirmStatusRight} />
    </box>
  )
}
