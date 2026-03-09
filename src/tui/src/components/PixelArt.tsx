import type { HalfBlockArt } from "../utils/pixelArt"

interface PixelArtProps {
  art: HalfBlockArt
  maxWidth?: number
  maxHeight?: number
}

export function PixelArt({ art, maxWidth, maxHeight }: PixelArtProps) {
  const visibleRows = maxHeight ? art.rows.slice(0, maxHeight) : art.rows
  const intrinsicWidth = visibleRows.reduce(
    (max, row) => Math.max(max, row.length),
    art.width,
  )
  const renderWidth = Math.max(
    0,
    maxWidth !== undefined ? Math.min(maxWidth, intrinsicWidth) : intrinsicWidth,
  )

  return (
    <box flexDirection="column" width={renderWidth} height={visibleRows.length}>
      {visibleRows.map((row, y) => {
        const cells = row.length >= renderWidth
          ? row.slice(0, renderWidth)
          : [
              ...row,
              ...Array.from({ length: renderWidth - row.length }, () => ({
                char: " ",
                fg: null,
                bg: null,
              })),
            ]

        return (
          <text key={y}>
            {cells.map((cell, x) => {
              if (cell.fg === null && cell.bg === null) {
                return cell.char
              }
              return (
                <span
                  key={x}
                  fg={cell.fg ?? undefined}
                  bg={cell.bg ?? undefined}
                >
                  {cell.char}
                </span>
              )
            })}
          </text>
        )
      })}
    </box>
  )
}
