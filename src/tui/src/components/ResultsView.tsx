import { useState, useMemo } from "react"
import { useKeyboard } from "@opentui/react"
import type { ComicResult } from "../types"
import { colors, confidenceColor } from "../theme"

interface ResultsViewProps {
  results: ComicResult[]
  onBack: () => void
}

export function ResultsView({ results, onBack }: ResultsViewProps) {
  const [search, setSearch] = useState("")
  const [filter, setFilter] = useState("all")
  const [focusSearch, setFocusSearch] = useState(false)

  const filterOptions = [
    { name: "All", description: "Show all results", value: "all" },
    { name: "High", description: "High confidence", value: "high" },
    { name: "Medium", description: "Medium confidence", value: "medium" },
    { name: "Low", description: "Low confidence", value: "low" },
  ]

  const filtered = useMemo(() => {
    return results.filter(r => {
      if (filter !== "all" && r.confidence !== filter) return false
      if (search && !r.title.toLowerCase().includes(search.toLowerCase())) return false
      return true
    })
  }, [results, filter, search])

  useKeyboard((key) => {
    if (key.name === "/" && !focusSearch) {
      setFocusSearch(true)
    }
    if (key.name === "escape") {
      if (focusSearch) {
        setFocusSearch(false)
      } else {
        onBack()
      }
    }
    if (key.name === "tab") {
      setFocusSearch(f => !f)
    }
  })

  // Column widths
  const titleW = 30
  const issueW = 8
  const yearW = 6
  const confW = 10

  return (
    <box flexDirection="column" flexGrow={1} paddingX={2} gap={1}>
      <text>
        <span fg={colors.white}>Scan Results</span>
        <span fg={colors.dimmed}>  ·  {filtered.length} comics</span>
      </text>

      <box flexDirection="row" gap={2}>
        <box flexDirection="row" gap={1}>
          <text fg={colors.dimmed}>Search:</text>
          <input
            value={search}
            onChange={setSearch}
            placeholder="filter by title..."
            focused={focusSearch}
            width={25}
          />
        </box>
        <tab-select
          options={filterOptions}
          focused={!focusSearch}
          onSelect={(_i, opt) => setFilter(opt?.value ?? "all")}
        />
      </box>

      {/* Table header */}
      <box flexDirection="row">
        <text fg={colors.dimmed} width={titleW}>{"Title".padEnd(titleW)}</text>
        <text fg={colors.dimmed} width={issueW}>{"Issue".padEnd(issueW)}</text>
        <text fg={colors.dimmed} width={yearW}>{"Year".padEnd(yearW)}</text>
        <text fg={colors.dimmed} width={confW}>{"Confidence".padEnd(confW)}</text>
      </box>

      {/* Table body */}
      <scrollbox flexGrow={1} height={15}>
        {filtered.map((r, i) => (
          <box key={i} flexDirection="row">
            <text width={titleW}>{r.title.slice(0, titleW - 1).padEnd(titleW)}</text>
            <text width={issueW}>{"#" + r.issue_number.padEnd(issueW - 1)}</text>
            <text width={yearW}>{r.year.padEnd(yearW)}</text>
            <text fg={confidenceColor(r.confidence)} width={confW}>{r.confidence}</text>
          </box>
        ))}
      </scrollbox>
    </box>
  )
}
