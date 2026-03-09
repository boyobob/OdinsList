import { useState, useEffect } from "react"

const RUNE_CHARS = ["\u16EF", "\u16F0", "\u16EA", "\u16E5", "\u16E9", "\u16DF", "\u16DE", "\u16CB", "\u16BB", "\u16A4"]
const RUNE_INTERVAL = 40

interface RuneSpinnerProps {
  color?: string
  playing?: boolean
}

export function RuneSpinner({ color = "#c8ccd4", playing = true }: RuneSpinnerProps) {
  const [frame, setFrame] = useState(0)

  useEffect(() => {
    if (!playing) return
    const interval = setInterval(() => {
      setFrame((f) => (f + 1) % RUNE_CHARS.length)
    }, RUNE_INTERVAL)
    return () => clearInterval(interval)
  }, [playing])

  return <text fg={color}>{RUNE_CHARS[frame]}</text>
}
