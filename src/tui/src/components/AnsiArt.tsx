import type { AnsiSegment } from "../utils/ansiArt"

interface AnsiArtProps {
  rows: AnsiSegment[][]
}

export function AnsiArt({ rows }: AnsiArtProps) {
  return (
    <box flexDirection="column" alignItems="center">
      {rows.map((row, y) => (
        <text key={y}>
          {row.map((segment, x) => {
            if (!segment.fg && !segment.bg) {
              return segment.text
            }
            return (
              <span
                key={x}
                fg={segment.fg ?? undefined}
                bg={segment.bg ?? undefined}
              >
                {segment.text}
              </span>
            )
          })}
        </text>
      ))}
    </box>
  )
}
