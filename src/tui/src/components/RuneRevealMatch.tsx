import { useState, useEffect, useRef } from "react"
import { colors, confidenceColor } from "../theme"

interface LatestMatchData {
  title: string
  issue: string
  month: string
  year: string
  publisher: string
  confidence: string
  source: string
  foundAtMs: number
}

interface BrowseData {
  title: string
  issue: string
  month: string
  year: string
  publisher: string
  confidence: string
  filename: string
  imagePath: string | null
}

interface RuneRevealMatchProps {
  match: LatestMatchData | null
  width: number
  mode: "live" | "browse"
  browseData: BrowseData | null
  selectedIndex: number
}

function parseHexColor(hex: string): [number, number, number] | null {
  const normalized = hex.trim().replace(/^#/, "")
  if (normalized.length !== 6) return null
  const value = Number.parseInt(normalized, 16)
  if (Number.isNaN(value)) return null
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255]
}

function blendHexColor(fromHex: string, toHex: string, ratio: number): string {
  const from = parseHexColor(fromHex)
  const to = parseHexColor(toHex)
  if (!from || !to) return fromHex
  const t = Math.max(0, Math.min(1, ratio))
  const r = Math.round(from[0] + ((to[0] - from[0]) * t))
  const g = Math.round(from[1] + ((to[1] - from[1]) * t))
  const b = Math.round(from[2] + ((to[2] - from[2]) * t))
  return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`
}

function formatIssue(issue: string): string {
  const normalized = issue.trim()
  if (!normalized) return "#?"
  if (normalized.startsWith("#")) return normalized
  return `#${normalized}`
}

function formatAge(foundAtMs: number, nowMs: number): string {
  const totalSeconds = Math.max(0, (nowMs - foundAtMs) / 1000)
  if (totalSeconds < 60) {
    return `${totalSeconds.toFixed(1)}s ago`
  }
  const minutes = Math.floor(totalSeconds / 60)
  const secs = totalSeconds - minutes * 60
  if (minutes < 60) {
    return `${minutes}m ${secs.toFixed(1)}s ago`
  }
  const hours = Math.floor(minutes / 60)
  const mins = minutes - hours * 60
  return `${hours}h ${mins}m ${Math.floor(secs)}s ago`
}

function confidenceBackground(level: string): string {
  if (level === "high") return "#0f3d0f"
  if (level === "medium") return "#5a3a12"
  if (level === "low") return "#4a1212"
  return colors.muted
}

function clip(value: string, maxWidth: number): string {
  if (maxWidth <= 0) return ""
  if (value.length <= maxWidth) return value
  if (maxWidth <= 3) return value.slice(0, maxWidth)
  return value.slice(0, maxWidth - 3) + "..."
}

// --- Rune Reveal Animation Frames ---
// Scattered braille particles coalesce into "Match Found!" with pulsing ⠶ borders
const REVEAL_TEXT = "Match Found!"
const PULSE_BORDER = "⠶⠶⠶"
const REVEAL_TO_PULSE_FRAME = 5

// Scattered particle characters (partial braille dots)
const SCATTER_CHARS = ["⠁", "⠂", "⠄", "⠈", "⠐", "⠠"]
// Coalescing characters (more dots filled in)
const COALESCE_CHARS = ["⠃", "⠆", "⠤", "⠴", "⠶"]

// --- Rune Browse Animation ---
const RUNE_CHARS = ["\u16EF", "\u16F0", "\u16EA", "\u16E5", "\u16E9", "\u16DF", "\u16DE", "\u16CB", "\u16BB", "\u16A4"]
const RUNE_FRAME_DURATION = 40
const RUNE_SLOT_WIDTH = 4  // " X " = padding + rune + padding + extra space for anchor

function generateRevealFrame(frameIndex: number, totalWidth: number): string {
  // Use the same anchor as pulse frames so there's no shift at the transition
  const widest = `⠆ ⠶⠶⠶${REVEAL_TEXT}⠶⠶⠶ ⠰`
  const padLeft = Math.max(0, Math.floor((totalWidth - widest.length) / 2))

  switch (frameIndex) {
    case 0: {
      // Sparse scattered dots
      const chars: string[] = []
      for (let i = 0; i < totalWidth; i++) {
        chars.push(Math.random() < 0.12 ? SCATTER_CHARS[Math.floor(Math.random() * SCATTER_CHARS.length)] : " ")
      }
      return chars.join("")
    }
    case 1: {
      // More scattered dots, slightly denser
      const chars: string[] = []
      for (let i = 0; i < totalWidth; i++) {
        chars.push(Math.random() < 0.2 ? SCATTER_CHARS[Math.floor(Math.random() * SCATTER_CHARS.length)] : " ")
      }
      return chars.join("")
    }
    case 2: {
      // Dots clustering toward center
      const chars: string[] = []
      const center = totalWidth / 2
      for (let i = 0; i < totalWidth; i++) {
        const distFromCenter = Math.abs(i - center) / center
        const prob = 0.3 * (1 - distFromCenter * 0.7)
        chars.push(Math.random() < prob ? COALESCE_CHARS[Math.floor(Math.random() * COALESCE_CHARS.length)] : " ")
      }
      return chars.join("")
    }
    case 3: {
      // Dense cluster at center, coalescing chars
      const chars: string[] = []
      const center = totalWidth / 2
      for (let i = 0; i < totalWidth; i++) {
        const distFromCenter = Math.abs(i - center) / center
        const prob = 0.5 * (1 - distFromCenter * 0.5)
        chars.push(Math.random() < prob ? COALESCE_CHARS[Math.floor(Math.random() * COALESCE_CHARS.length)] : " ")
      }
      return chars.join("")
    }
    case 4: {
      // Text almost formed, ⠶ dots merging
      const raw = `${" ".repeat(padLeft)}  ⠶⠶⠶${REVEAL_TEXT}⠶⠶⠶  `
      const chars = raw.padEnd(totalWidth).split("")
      for (let i = 0; i < totalWidth; i++) {
        if (chars[i] === " " && Math.random() < 0.15) {
          chars[i] = COALESCE_CHARS[Math.floor(Math.random() * COALESCE_CHARS.length)]
        }
      }
      return chars.join("")
    }
    case 5: {
      // Match pulse frame 5 so the later handoff can be seamless.
      return generatePulseFrame(5, totalWidth)
    }
    case 6: {
      // Match pulse frame 4 (widest point).
      return generatePulseFrame(4, totalWidth)
    }
    case 7: {
      // Final reveal frame is identical to the first pulse frame at handoff.
      return generatePulseFrame(REVEAL_TO_PULSE_FRAME, totalWidth)
    }
    default:
      return `${" ".repeat(padLeft)}  ⠶⠶⠶${REVEAL_TEXT}⠶⠶⠶  `
  }
}

// Pulse frames (steady state, matches the user's JSON animation style)
const PULSE_FRAME_COUNT = 8

function generatePulseFrame(frameIndex: number, totalWidth: number): string {
  // All frames are built relative to a fixed anchor so the core text never shifts.
  // The widest frame is: "⠆ ⠶⠶⠶Match Found!⠶⠶⠶ ⠰" (frame 4).
  // We center based on that widest width, then pad each narrower frame symmetrically.
  const widest = `⠆ ⠶⠶⠶${REVEAL_TEXT}⠶⠶⠶ ⠰`
  const anchorPad = Math.max(0, Math.floor((totalWidth - widest.length) / 2))
  const frame = frameIndex % PULSE_FRAME_COUNT

  // Each frame is defined as [leftDecor, rightDecor] around REVEAL_TEXT.
  // The space budget on each side from the anchor is 5 chars ("⠆ ⠶⠶⠶").
  const frames: [string, string][] = [
    ["    ⠶", "⠶    "],  // 0
    ["   ⠶⠶", "⠶⠶   "],  // 1
    ["  ⠶⠶⠶", "⠶⠶⠶  "],  // 2
    [" ⠆⠶⠶⠶", "⠶⠶⠶⠰ "],  // 3
    ["⠆ ⠶⠶⠶", "⠶⠶⠶ ⠰"],  // 4
    [" ⠆⠶⠶⠶", "⠶⠶⠶⠰ "],  // 5
    ["  ⠶⠶⠶", "⠶⠶⠶  "],  // 6
    ["   ⠶⠶", "⠶⠶   "],  // 7
  ]

  const [left, right] = frames[frame] ?? frames[2]
  return `${" ".repeat(anchorPad)}${left}${REVEAL_TEXT}${right}`
}

type AnimPhase = "idle" | "reveal" | "pulse"

export function RuneRevealMatch({ match, width, mode, browseData, selectedIndex }: RuneRevealMatchProps) {
  const [phase, setPhase] = useState<AnimPhase>("idle")
  const [revealFrame, setRevealFrame] = useState(0)
  const [pulseFrame, setPulseFrame] = useState(0)
  const [nowMs, setNowMs] = useState(Date.now())
  const [pulsePhase, setPulsePhase] = useState(0)
  const prevMatchRef = useRef<LatestMatchData | null>(null)
  // Cache reveal frames so they don't re-randomize on every render
  const revealFramesRef = useRef<string[]>([])
  const contentWidth = Math.max(12, width - 2)

  // --- Rune browse animation state ---
  const [runeFrame, setRuneFrame] = useState(-1) // -1 = idle (show spaces)
  const [runeKey, setRuneKey] = useState(0) // increment to force effect restart
  const prevSelectedIndexRef = useRef<number>(-2) // start different from any real index

  // Restart rune animation when selectedIndex changes in browse mode
  useEffect(() => {
    if (mode !== "browse") return
    if (selectedIndex === prevSelectedIndexRef.current) return
    prevSelectedIndexRef.current = selectedIndex
    if (selectedIndex < 0) return
    setRuneFrame(0)
    setRuneKey((k) => k + 1)
  }, [mode, selectedIndex])

  // Rune animation timer
  useEffect(() => {
    if (runeFrame < 0 || runeFrame >= RUNE_CHARS.length) return
    const timer = setTimeout(() => {
      const next = runeFrame + 1
      if (next >= RUNE_CHARS.length) {
        setRuneFrame(-1) // animation complete, show spaces
      } else {
        setRuneFrame(next)
      }
    }, RUNE_FRAME_DURATION)
    return () => clearTimeout(timer)
  }, [runeFrame, runeKey])

  // When a new match arrives, trigger the reveal animation
  useEffect(() => {
    if (!match) return
    if (prevMatchRef.current?.foundAtMs === match.foundAtMs) return
    prevMatchRef.current = match
    // Pre-generate all reveal frames so randomness is fixed per reveal
    const frames: string[] = []
    for (let i = 0; i < 8; i++) {
      frames.push(generateRevealFrame(i, contentWidth))
    }
    revealFramesRef.current = frames
    setRevealFrame(0)
    setPhase("reveal")
    setPulsePhase(0)
  }, [match, width])

  // Reveal animation timer
  useEffect(() => {
    if (phase !== "reveal") return
    const interval = setInterval(() => {
      setRevealFrame((f) => {
        if (f >= 7) {
          setPhase("pulse")
          setPulseFrame(REVEAL_TO_PULSE_FRAME)
          return 7
        }
        return f + 1
      })
    }, 125)
    return () => clearInterval(interval)
  }, [phase])

  // Pulse animation timer (steady state — live mode only)
  useEffect(() => {
    if (phase !== "pulse") return
    const interval = setInterval(() => {
      setPulseFrame((f) => (f + 1) % PULSE_FRAME_COUNT)
    }, 125)
    return () => clearInterval(interval)
  }, [phase])

  // Color pulse phase (sine wave for text color — live mode only)
  useEffect(() => {
    if (!match) return
    const interval = setInterval(() => {
      setPulsePhase((p) => (p + 0.18) % (Math.PI * 2))
    }, 80)
    return () => clearInterval(interval)
  }, [match])

  // Age timer
  useEffect(() => {
    if (!match) return
    const interval = setInterval(() => {
      setNowMs(Date.now())
    }, 100)
    return () => clearInterval(interval)
  }, [match])

  // Colors
  const baseColor = match ? confidenceColor(match.confidence) : colors.dimmed
  const pulseMix = 0.08 + (((Math.sin(pulsePhase) + 1) / 2) * 0.18)
  const textColor = match ? blendHexColor(baseColor, colors.white, pulseMix) : colors.dimmed
  const bannerColor = blendHexColor(colors.accent, colors.white, pulseMix + 0.1)

  // Browse mode render
  if (mode === "browse") {
    if (!browseData) {
      return (
        <box flexDirection="column" alignItems="center" justifyContent="center">
          <text fg={colors.dimmed}>{"Select a result to preview..."}</text>
        </box>
      )
    }

    const title = browseData.title || "(untitled)"
    const issue = formatIssue(browseData.issue)
    const month = (browseData.month || "UNK").toUpperCase()
    const year = browseData.year || "????"
    const publisher = browseData.publisher || "Unknown"
    const confLabel = ` ${browseData.confidence} `
    const confBg = confidenceBackground(browseData.confidence)

    const browseInfoLine = `${title} \u2022 ${issue} \u2022 ${month} \u2022 ${year} \u2022 ${publisher}`
    const browseBaseColor = confidenceColor(browseData.confidence)

    // Rune slots: fixed width, either show rune or space
    const runeChar = runeFrame >= 0 && runeFrame < RUNE_CHARS.length ? RUNE_CHARS[runeFrame] : " "
    const leftRune = ` ${runeChar} `
    const rightRune = ` ${runeChar} `
    const runeColor = runeFrame >= 0 ? browseBaseColor : colors.dimmed

    // Clip info line accounting for rune slots + confidence badge
    const availableForInfo = contentWidth - (RUNE_SLOT_WIDTH * 2) - confLabel.length - 2
    const clippedInfo = clip(browseInfoLine, availableForInfo)

    // Banner: static centered "Match Result"
    const browseLabel = "⠆  Match Result  ⠰"
    const browsePadLeft = Math.max(0, Math.floor((contentWidth - browseLabel.length) / 2))
    const browseBannerLine = " ".repeat(browsePadLeft) + browseLabel
    const browseBannerColor = colors.accent

    // Bottom row: abbreviated path (last 2 segments) left, filename right
    const filename = browseData.filename || ""
    const abbrevPath = browseData.imagePath
      ? browseData.imagePath.split("/").slice(-3, -1).join("/")
      : ""
    const pathPrefix = abbrevPath ? `~/${abbrevPath}` : ""

    return (
      <box flexDirection="column">
        <text fg={browseBannerColor}>{browseBannerLine}</text>
        <text> </text>
        <box flexDirection="row" justifyContent="center">
          <text fg={runeColor}>{leftRune}</text>
          <text fg={browseBaseColor}>{clippedInfo}</text>
          <text>{"  "}</text>
          <text fg={colors.white} bg={confBg}>{confLabel}</text>
          <text fg={runeColor}>{rightRune}</text>
        </box>
        <text> </text>
        <box flexDirection="row" justifyContent="space-between">
          <text fg={colors.dimmed}>{clip(pathPrefix, Math.floor(contentWidth / 2))}</text>
          <text fg={colors.dimmed}>{clip(filename, Math.floor(contentWidth / 2))}</text>
        </box>
      </box>
    )
  }

  if (!match) {
    return (
      <box flexDirection="column" alignItems="center" justifyContent="center">
        <text fg={colors.dimmed}>{"Waiting for Match..."}</text>
      </box>
    )
  }

  // Banner line (animated)
  let bannerLine: string
  if (phase === "reveal") {
    bannerLine = revealFramesRef.current[revealFrame] ?? generateRevealFrame(revealFrame, contentWidth)
  } else if (phase === "pulse") {
    bannerLine = generatePulseFrame(pulseFrame, contentWidth)
  } else {
    bannerLine = generatePulseFrame(0, contentWidth)
  }

  // Info lines
  const title = match.title || "(untitled)"
  const issue = formatIssue(match.issue)
  const month = (match.month || "UNK").toUpperCase()
  const year = match.year || "????"
  const publisher = match.publisher || "Unknown"
  const confLabel = ` ${match.confidence} `
  const confBg = confidenceBackground(match.confidence)

  const infoLine = `${title} \u2022 ${issue} \u2022 ${month} \u2022 ${year} \u2022 ${publisher}`
  const sourceLine = `Source: ${match.source}`
  const ageLine = formatAge(match.foundAtMs, nowMs)

  // Right-align age on the second info line
  const sourceLineWidth = sourceLine.length
  const ageWidth = ageLine.length
  const gap = Math.max(2, contentWidth - sourceLineWidth - ageWidth)
  const bottomLine = `${sourceLine}${" ".repeat(gap)}${ageLine}`

  return (
    <box flexDirection="column">
      <text fg={bannerColor}>{bannerLine}</text>
      <text> </text>
      <box flexDirection="row" justifyContent="center">
        <text fg={textColor}>{clip(infoLine, contentWidth - confLabel.length - 2)}</text>
        <text>{"  "}</text>
        <text fg={colors.white} bg={confBg}>{confLabel}</text>
      </box>
      <text> </text>
      <box flexDirection="row" justifyContent="center">
        <text fg={colors.dimmed}>{clip(bottomLine, contentWidth)}</text>
      </box>
    </box>
  )
}
