import { describe, it, expect } from "bun:test"
import { parseAnsiArt } from "./ansiArt"

describe("parseAnsiArt", () => {
  it("parses a simple segment with fg color", () => {
    const input = String.raw`\e[38;5;233m▄\e[m`
    const result = parseAnsiArt(input)
    expect(result).toHaveLength(1) // 1 line
    expect(result[0]).toContainEqual({ text: "▄", fg: "#121212", bg: undefined })
  })

  it("parses fg + bg colors", () => {
    const input = String.raw`\e[38;5;24;48;5;17m▄▄\e[m`
    const result = parseAnsiArt(input)
    expect(result[0]).toContainEqual({ text: "▄▄", fg: "#005f87", bg: "#00005f" })
  })

  it("handles reset \\e[m and \\e[49m (default bg)", () => {
    const input = String.raw`\e[49m   \e[38;5;233m▄\e[m`
    const result = parseAnsiArt(input)
    expect(result[0]).toContainEqual({ text: "   ", fg: undefined, bg: undefined })
    expect(result[0]).toContainEqual({ text: "▄", fg: "#121212", bg: undefined })
  })

  it("splits lines on newline", () => {
    const input = String.raw`\e[49m▄\e[m` + "\n" + String.raw`\e[49m▄\e[m`
    const result = parseAnsiArt(input)
    expect(result).toHaveLength(2)
  })

  it("handles combined fg;bg in one sequence like 38;5;N;48;5;N", () => {
    const input = String.raw`\e[38;5;67;48;5;74m▄\e[m`
    const result = parseAnsiArt(input)
    expect(result[0]).toContainEqual({ text: "▄", fg: "#5f87af", bg: "#5fafd7" })
  })

  it("strips printf wrapper if present", () => {
    const input = `printf "\\e[49m▄\\e[m";`
    const result = parseAnsiArt(input)
    expect(result[0].some(s => s.text === "▄")).toBe(true)
  })

  it("merges consecutive segments with same colors", () => {
    const input = String.raw`\e[38;5;233m▄\e[38;5;233m▄\e[m`
    const result = parseAnsiArt(input)
    expect(result[0]).toContainEqual({ text: "▄▄", fg: "#121212", bg: undefined })
  })

  it("strips trailing blank lines", () => {
    const input = String.raw`\e[38;5;233m▄\e[m` + "\n" + String.raw`\e[49m                    \e[m`
    const result = parseAnsiArt(input)
    expect(result).toHaveLength(1)
  })
})
