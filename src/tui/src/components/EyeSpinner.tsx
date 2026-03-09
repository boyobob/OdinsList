import { useState, useEffect } from "react"
import { colors } from "../theme"

type EyeFrame = [string, string, string]

// Keep the art in a slightly wider slot so the animation sits farther left
// relative to the label and doesn't feel like it jitters toward the text.
export const EYE_SPINNER_WIDTH = 9
export const EYE_SPINNER_HEIGHT = 3

// Frames from /home/bobby/Downloads/Untitled Project(1).txt anchored so the
// "0" stays on the middle row, which is the row aligned with the menu label.
const SEQUENCE: EyeFrame[] = [
  [" ⠣   ⠜", "   0", ""],
  ["", "⠤⠤ 0 ⠤⠤", ""],
  ["", "   0", " ⠎   ⠱"],
  ["", "⠤⠤ 0 ⠤⠤", ""],
]

export function EyeSpinner() {
  const [frameIndex, setFrameIndex] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setFrameIndex(i => (i + 1) % SEQUENCE.length)
    }, 200)

    return () => clearInterval(interval)
  }, [])

  return (
    <box flexDirection="column" width={EYE_SPINNER_WIDTH} height={EYE_SPINNER_HEIGHT}>
      {SEQUENCE[frameIndex].map((row, i) => (
        <text key={i} fg={colors.accent}>
          {row.padEnd(EYE_SPINNER_WIDTH)}
        </text>
      ))}
    </box>
  )
}
