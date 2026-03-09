// src/tui/src/components/StatusBar.tsx
import { useTerminalDimensions } from "@opentui/react"
import { colors } from "../theme"

interface StatusBarProps {
  left: string
  right: string
}

function truncateHint(text: string, maxWidth: number): string {
  if (maxWidth <= 0) return ""
  if (text.length <= maxWidth) return text
  if (maxWidth === 1) return "…"
  return `${text.slice(0, maxWidth - 1)}…`
}

export function StatusBar({ left, right }: StatusBarProps) {
  const { width } = useTerminalDimensions()
  const horizontalPadding = 2
  const contentWidth = Math.max(0, width - (horizontalPadding * 2))
  const hasBoth = left.length > 0 && right.length > 0
  const minGap = hasBoth ? 2 : 0
  const fitsSingleLine = !hasBoth || (left.length + right.length + minGap <= contentWidth)

  if (!hasBoth || fitsSingleLine) {
    const leftText = truncateHint(left, contentWidth)
    const rightText = truncateHint(right, contentWidth)
    return (
      <box
        flexDirection="row"
        justifyContent="space-between"
        paddingX={horizontalPadding}
        height={1}
      >
        <text fg={colors.statusHint}>{leftText}</text>
        <text fg={colors.statusHint}>{rightText}</text>
      </box>
    )
  }

  const leftText = truncateHint(left, contentWidth)
  const rightText = truncateHint(right, contentWidth)

  return (
    <box
      flexDirection="column"
      paddingX={horizontalPadding}
      height={2}
    >
      <text fg={colors.statusHint}>{leftText}</text>
      <box flexDirection="row" justifyContent="flex-end">
        <text fg={colors.statusHint}>{rightText}</text>
      </box>
    </box>
  )
}
