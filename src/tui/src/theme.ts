export const colors = {
  bg: "#0a0a0f",
  fg: "#c8ccd4",
  accent: "#6e8ab5",
  dimmed: "#899cc7",
  statusHint: "#c8ccd4",
  highlight: "#ffffff",
  green: "#6a9955",
  yellow: "#c4a24d",
  red: "#c24d4d",
  cyan: "#5e9ea0",
  magenta: "#8a6eaf",
  muted: "#2a2e38",
  white: "#c8ccd4",
} as const

export function confidenceColor(level: string): string {
  switch (level) {
    case "high": return colors.green
    case "medium": return colors.yellow
    case "low": return colors.red
    default: return colors.dimmed
  }
}

export function tagColor(tag: string): string {
  switch (tag) {
    case "VLM": return colors.cyan
    case "GCD": return colors.yellow
    case "COMICVINE": return colors.magenta
    case "RESULT": return colors.green
    case "ERROR": return colors.red
    case "BOX": return colors.accent
    default: return colors.dimmed
  }
}
