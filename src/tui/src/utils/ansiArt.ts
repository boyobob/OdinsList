import { readFileSync } from "fs"

export interface AnsiSegment {
  text: string
  fg: string | undefined
  bg: string | undefined
}

// 256-color palette: indices 0-255 -> hex RGB
const PALETTE: string[] = buildPalette()

function buildPalette(): string[] {
  const p: string[] = []
  // 0-7: standard colors
  const std = ["#000000","#800000","#008000","#808000","#000080","#800080","#008080","#c0c0c0"]
  // 8-15: bright colors
  const bright = ["#808080","#ff0000","#00ff00","#ffff00","#0000ff","#ff00ff","#00ffff","#ffffff"]
  p.push(...std, ...bright)
  // 16-231: 6x6x6 color cube
  const vals = [0, 0x5f, 0x87, 0xaf, 0xd7, 0xff]
  for (let r = 0; r < 6; r++)
    for (let g = 0; g < 6; g++)
      for (let b = 0; b < 6; b++)
        p.push(`#${hex(vals[r])}${hex(vals[g])}${hex(vals[b])}`)
  // 232-255: grayscale ramp
  for (let i = 0; i < 24; i++) {
    const v = 8 + i * 10
    p.push(`#${hex(v)}${hex(v)}${hex(v)}`)
  }
  return p
}

function hex(n: number): string {
  return n.toString(16).padStart(2, "0")
}

/**
 * Parse ANSI art from a string containing literal \e[...m sequences
 * (as found in printf-style files). Returns rows of colored segments.
 */
export function parseAnsiArt(raw: string): AnsiSegment[][] {
  // Unescape \uNNNN sequences (Bun's String.raw escapes non-ASCII chars)
  let content = raw.replace(/\\u([0-9a-fA-F]{4})/g, (_, h) =>
    String.fromCharCode(parseInt(h, 16))
  )
  if (content.startsWith('printf "')) {
    content = content.slice('printf "'.length)
    // Remove trailing ";
    const endIdx = content.lastIndexOf('";')
    if (endIdx !== -1) content = content.substring(0, endIdx)
  }

  const lines = content.split("\n")
  const rows = lines.map(line => parseAnsiLine(line))

  // Strip trailing rows that contain only whitespace
  while (rows.length > 0) {
    const last = rows[rows.length - 1]
    const allBlank = last.every(s => s.text.trim() === "")
    if (allBlank) rows.pop()
    else break
  }

  return rows
}

function parseAnsiLine(line: string): AnsiSegment[] {
  const segments: AnsiSegment[] = []
  let fg: string | undefined
  let bg: string | undefined

  // Match literal \e[ sequences (not real ESC bytes)
  const regex = /\\e\[([^\\]*?)m|([^\\]+|\\(?!e\[))/g
  let match: RegExpExecArray | null

  while ((match = regex.exec(line)) !== null) {
    if (match[1] !== undefined) {
      // It's an escape sequence — parse the SGR codes
      const codes = match[1]
      if (codes === "" || codes === "0") {
        fg = undefined
        bg = undefined
      } else {
        const parts = codes.split(";").map(Number)
        let i = 0
        while (i < parts.length) {
          if (parts[i] === 38 && parts[i + 1] === 5) {
            fg = PALETTE[parts[i + 2]] ?? undefined
            i += 3
          } else if (parts[i] === 48 && parts[i + 1] === 5) {
            bg = PALETTE[parts[i + 2]] ?? undefined
            i += 3
          } else if (parts[i] === 49) {
            bg = undefined
            i++
          } else if (parts[i] === 39) {
            fg = undefined
            i++
          } else if (parts[i] === 0) {
            fg = undefined
            bg = undefined
            i++
          } else {
            i++
          }
        }
      }
    } else if (match[2] !== undefined) {
      // It's visible text
      const text = match[2]
      if (text.length > 0) {
        // Merge with previous segment if same colors
        const prev = segments.length > 0 ? segments[segments.length - 1] : null
        if (prev && prev.fg === fg && prev.bg === bg) {
          prev.text += text
        } else {
          segments.push({ text, fg, bg })
        }
      }
    }
  }

  return segments
}

/**
 * Load and parse an ANSI art file from disk.
 */
export function loadAnsiArt(path: string): AnsiSegment[][] {
  const raw = readFileSync(path, "utf-8").trimEnd()
  return parseAnsiArt(raw)
}
