import { useState } from "react"
import { useKeyboard, useRenderer, useTerminalDimensions } from "@opentui/react"
import { MainMenu } from "./MainMenu"
import type { MenuItem } from "./MainMenu"
import { StatusBar } from "./StatusBar"
import { AnsiArt } from "./AnsiArt"
import { colors } from "../theme"
import type { AnsiSegment } from "../utils/ansiArt"

interface HomeViewProps {
  configReady: boolean
  eyeLogoRows: AnsiSegment[][]
  textLogoRaw: string
  configErrors: string[]
  configWarnings: string[]
  onStartNewRun: () => void
  onResumeRun: () => void
  onSettings: () => void
}

export function HomeView({ configReady, eyeLogoRows, textLogoRaw, configErrors, configWarnings, onStartNewRun, onResumeRun, onSettings }: HomeViewProps) {
  const renderer = useRenderer()
  const { width, height } = useTerminalDimensions()
  const [showHelp, setShowHelp] = useState(false)

  const EYE_LOGO_WIDTH = Math.max(
    ...eyeLogoRows.map(row => row.reduce((w, seg) => w + seg.text.length, 0))
  )

  const showEyeLogo = width >= EYE_LOGO_WIDTH + 2
  const compact = height < 25

  const items: MenuItem[] = [
    { label: configReady ? "Start New Run" : "Start New Run (loading...)", value: "new_run" },
    { label: configReady ? "Resume Run" : "Resume Run (loading...)", value: "resume" },
    { label: configReady ? "Settings/Config" : "Settings/Config (loading...)", value: "settings" },
    { label: "Quit", value: "quit" },
  ]

  const handleSelect = (item: MenuItem) => {
    switch (item.value) {
      case "new_run":
        if (configReady) onStartNewRun()
        break
      case "resume":
        if (configReady) onResumeRun()
        break
      case "settings":
        if (configReady) onSettings()
        break
      case "quit": renderer.destroy(); break
    }
  }

  useKeyboard((key) => {
    if (key.name === "h") {
      setShowHelp(v => !v)
      return
    }
    if (key.name === "escape" && showHelp) {
      setShowHelp(false)
    }
  })

  return (
    <box flexDirection="column" width="100%" height="100%">
      <box flexDirection="column" alignItems="center" justifyContent="center" flexGrow={1}>
        {showEyeLogo && !compact && (
          <AnsiArt rows={eyeLogoRows} />
        )}
        <text fg={colors.fg}>{textLogoRaw}</text>
        {(configErrors.length > 0 || configWarnings.length > 0) && (
          <box flexDirection="column" marginTop={1} paddingX={2}>
            {configErrors.map((err, i) => (
              <text key={`err-${i}`} fg={colors.red}>  ! {err}</text>
            ))}
            {configWarnings.map((warn, i) => (
              <text key={`warn-${i}`} fg={colors.red}>  ! {warn}</text>
            ))}
          </box>
        )}
        {showHelp ? (
          <box flexDirection="column" marginTop={compact ? 0 : 1} border borderStyle="single" paddingX={2} paddingY={1} gap={1}>
            <text fg={colors.white}><strong>Keyboard Help</strong></text>
            <text fg={colors.dimmed}>[↑/↓] Navigate Menu Items</text>
            <text fg={colors.dimmed}>[Enter] Select Menu Item</text>
            <text fg={colors.dimmed}>[H] Toggle Keyboard Help</text>
            <text fg={colors.dimmed}>[Q] Quit</text>
            <text fg={colors.dimmed}>[Esc] Back from Sub-Screens</text>
            <text fg={colors.dimmed}>[Ctrl+S] Save Settings/Config Changes</text>
            <text fg={colors.dimmed}>[Ctrl+Shift+V] Paste into Text Inputs</text>
            <text fg={colors.accent}>[H] or [Esc] Close This Help Panel</text>
          </box>
        ) : (
          <box marginTop={compact ? 0 : 1}>
            <MainMenu
              items={items}
              onSelect={handleSelect}
            />
          </box>
        )}
      </box>
      <StatusBar
        left={showHelp ? "[H]/[Esc] Close Help" : "[↑/↓] Navigate  ⋄  [Enter] Select"}
        right={showHelp ? "[Q] Quit" : "[H] Help  ⋄  [Q] Quit"}
      />
    </box>
  )
}
