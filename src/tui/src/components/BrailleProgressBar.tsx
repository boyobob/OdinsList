import { useState, useEffect } from "react"
import { colors } from "../theme"

interface BrailleProgressBarProps {
  progress: number  // 0-100
  width?: number
  playing?: boolean
}

// Full braille = U+28FF, empty braille = U+2800
// Shimmer sequence for leading edge (partially filled -> full)
const SHIMMER_CHARS = ["\u28F7", "\u28EF", "\u28DF", "\u28FF"]
const SHIMMER_INTERVAL = 150

export function BrailleProgressBar({ progress, width = 40, playing = true }: BrailleProgressBarProps) {
  const [shimmerIndex, setShimmerIndex] = useState(0)

  useEffect(() => {
    if (!playing) return
    const interval = setInterval(() => {
      setShimmerIndex((i) => (i + 1) % SHIMMER_CHARS.length)
    }, SHIMMER_INTERVAL)
    return () => clearInterval(interval)
  }, [playing])

  const clamped = Math.max(0, Math.min(100, progress))
  const filledExact = (clamped / 100) * width
  const filledFull = Math.floor(filledExact)
  const hasEdge = filledFull < width && clamped > 0 && clamped < 100
  const emptyCount = Math.max(0, width - filledFull - (hasEdge ? 1 : 0))

  const filled = "\u28FF".repeat(filledFull)
  const edge = hasEdge ? (playing ? SHIMMER_CHARS[shimmerIndex] : SHIMMER_CHARS[SHIMMER_CHARS.length - 1]) : ""
  const empty = "\u2800".repeat(emptyCount)

  const bar = `${filled}${edge}${empty}`
  const pct = `${Math.round(clamped)}%`

  return (
    <box flexDirection="row" gap={1}>
      <text fg={colors.accent}>{bar}</text>
      <text fg={colors.dimmed}>{pct}</text>
    </box>
  )
}
