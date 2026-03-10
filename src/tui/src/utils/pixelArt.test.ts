import { describe, expect, test } from "bun:test"
import { PNG } from "pngjs"
import { convertPngToHalfBlock } from "./pixelArt"

function buildPng(width: number, height: number, pixels: Array<[number, number, number, number]>): Buffer {
  const png = new PNG({ width, height })

  for (let index = 0; index < pixels.length; index += 1) {
    const [r, g, b, a] = pixels[index] ?? [0, 0, 0, 0]
    const offset = index * 4
    png.data[offset] = r
    png.data[offset + 1] = g
    png.data[offset + 2] = b
    png.data[offset + 3] = a
  }

  return PNG.sync.write(png)
}

describe("convertPngToHalfBlock", () => {
  test("preserves top and bottom colors in a half-block cell", () => {
    const buffer = buildPng(1, 2, [
      [255, 0, 0, 255],
      [0, 0, 255, 255],
    ])

    const art = convertPngToHalfBlock(buffer)

    expect(art.rows).toHaveLength(1)
    expect(art.rows[0]).toEqual([{ char: "▀", fg: "#ff0000", bg: "#0000ff" }])
  })

  test("collapses matching colors into a full block", () => {
    const buffer = buildPng(1, 2, [
      [12, 34, 56, 255],
      [12, 34, 56, 255],
    ])

    const art = convertPngToHalfBlock(buffer)

    expect(art.rows[0]).toEqual([{ char: "█", fg: "#0c2238", bg: null }])
  })
})
