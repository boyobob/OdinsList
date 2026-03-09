import { useCallback, useEffect, useRef, useState } from "react"
import { useRenderer } from "@opentui/react"
import type { Config } from "../types"
import { colors } from "../theme"

interface SettingsViewProps {
  config: Config
  onSave: (config: Config) => void
  onCancel: () => void
}

const fields = [
  { key: "vlm_base_url", label: "VLM API URL" },
  { key: "vlm_model", label: "VLM Model" },
  { key: "comicvine_api_key", label: "ComicVine API Key" },
  { key: "gcd_db_path", label: "GCD Database Path" },
] as const

const toggles = [
  { key: "gcd_enabled", label: "GCD Search" },
  { key: "comicvine_enabled", label: "ComicVine Lookup" },
] as const

export function SettingsView({ config, onSave, onCancel }: SettingsViewProps) {
  const renderer = useRenderer()
  const [draft, setDraft] = useState<Config>({ ...config })
  const [focusIndex, setFocusIndex] = useState(0)
  const draftRef = useRef<Config>({ ...config })

  const totalFields = fields.length + toggles.length
  const saveKeyBindings = [
    { name: "return", action: "newline" as const },
    { name: "linefeed", action: "newline" as const },
    { name: "s", ctrl: true, action: "submit" as const },
  ]

  useEffect(() => {
    draftRef.current = draft
  }, [draft])

  const saveDraft = useCallback(() => {
    onSave(draftRef.current)
  }, [onSave])

  const updateDraftField = useCallback((key: keyof Config, value: string | boolean) => {
    const next = { ...draftRef.current, [key]: value } as Config
    draftRef.current = next
    setDraft(next)
  }, [])

  useEffect(() => {
    const handler = (key: any) => {
      if (key.eventType === "release") return

      if (key.name === "tab") {
        if (key.shift) setFocusIndex(i => (i - 1 + totalFields) % totalFields)
        else setFocusIndex(i => (i + 1) % totalFields)
        return
      }

      if (key.ctrl && key.name === "s") {
        saveDraft()
        return
      }

      if (key.name === "escape") {
        onCancel()
        return
      }

      if ((key.name === "enter" || key.name === "return" || key.name === "space") && focusIndex >= fields.length) {
        const toggleIdx = focusIndex - fields.length
        const toggleKey = toggles[toggleIdx]?.key
        if (!toggleKey) return
        updateDraftField(toggleKey, !draftRef.current[toggleKey])
      }
    }

    renderer.keyInput.on("keypress", handler)
    return () => { renderer.keyInput.off("keypress", handler) }
  }, [renderer, totalFields, focusIndex, saveDraft, onCancel, updateDraftField])

  return (
    <box flexDirection="column" flexGrow={1} paddingX={2} gap={1}>
      <text fg={colors.white}>Settings</text>

      <box flexDirection="column" gap={1} marginTop={1}>
        {fields.map((field, i) => (
          <box key={field.key} flexDirection="row" gap={1}>
            <text fg={colors.dimmed} width={20}>{field.label}</text>
            <input
              value={(draft as any)[field.key]}
              onInput={(v) => updateDraftField(field.key, v)}
              keyBindings={saveKeyBindings}
              onSubmit={saveDraft}
              focused={focusIndex === i}
              width={40}
            />
          </box>
        ))}
      </box>

      <box flexDirection="column" gap={1} marginTop={1}>
        {toggles.map((toggle, i) => {
          const checked = (draft as any)[toggle.key]
          const idx = fields.length + i
          return (
            <box key={toggle.key} flexDirection="row" gap={1}>
              <text fg={focusIndex === idx ? colors.accent : colors.dimmed}>
                {checked ? "[x]" : "[ ]"}
              </text>
              <text fg={focusIndex === idx ? colors.white : colors.dimmed}>
                {toggle.label}
              </text>
            </box>
          )
        })}
      </box>
    </box>
  )
}
