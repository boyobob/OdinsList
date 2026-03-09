import { useState, useEffect, useCallback } from "react"
import { useKeyboard, useRenderer } from "@opentui/react"
import { useBackend } from "./hooks/useBackend"
import { useAppState } from "./hooks/useAppState"
import { StatusBar } from "./components/StatusBar"
import { HomeView } from "./components/HomeView"
import { NewRunWizard } from "./components/NewRunWizard"
import { ActiveRunView } from "./components/ActiveRunView"
import { ResultsView } from "./components/ResultsView"
import { SettingsView } from "./components/SettingsView"
import type { Config, ComicResult, ScanEvent, RunMode, ResumeStats } from "./types"
import { colors } from "./theme"
import { parseAnsiArt } from "./utils/ansiArt"
import { EYE_LOGO_RAW, TEXT_LOGO_RAW } from "./embeddedAssets"

// Parse the embedded logo once at startup (shared between HomeView and NewRunWizard)
const EYE_LOGO_ROWS = parseAnsiArt(EYE_LOGO_RAW)

const statusBarContent: Record<string, { left: string; right: string }> = {
  results: { left: "/: Search  ⋄  Tab: Toggle Focus", right: "Esc: Back" },
  settings: {
    left: "[Tab/Shift+Tab] Navigate  ⋄  [Enter/Space] Toggle",
    right: "[Ctrl+S] Save  ⋄  [Ctrl+Shift+V] Paste  ⋄  [Esc] Cancel",
  },
}

export function App() {
  const renderer = useRenderer()
  const backend = useBackend()
  const { state, go, back } = useAppState("home")

  const [config, setConfig] = useState<Config | null>(null)
  const [results, setResults] = useState<ComicResult[]>([])
  const [configErrors, setConfigErrors] = useState<string[]>([])
  const [configWarnings, setConfigWarnings] = useState<string[]>([])
  const [activeRunAutoStart, setActiveRunAutoStart] = useState(false)
  const [activeRunResumeMode, setActiveRunResumeMode] = useState(false)
  const [wizardMode, setWizardMode] = useState<"new" | "resume">("new")
  const [activeRunContext, setActiveRunContext] = useState<{
    inputRootDir: string
    outputTsvPath: string
    runMode: RunMode
    singleBoxDir: string
    resumeStats: ResumeStats | null
  }>({
    inputRootDir: "",
    outputTsvPath: "",
    runMode: "batch",
    singleBoxDir: "",
    resumeStats: null,
  })

  // Fetch config on startup
  useEffect(() => {
    if (!backend.ready) return

    const unsub = backend.onEvent((event: ScanEvent) => {
      if (event.event === "config") {
        setConfig(event.config)
      }
      if (event.event === "ScanComplete") {
        setResults(prev => [...prev, event.result])
      }
      if (event.event === "config_validation") {
        setConfigErrors(event.errors)
        setConfigWarnings(event.warnings)
      }
    })

    backend.sendCommand({ cmd: "get-config" })
    backend.sendCommand({ cmd: "validate-config" })
    return unsub
  }, [backend.ready])

  // Re-validate config when returning to home
  useEffect(() => {
    if (state === "home" && backend.ready) {
      backend.sendCommand({ cmd: "validate-config" })
    }
  }, [state, backend.ready])

  // Global keyboard shortcuts (only for states that use them)
  useKeyboard((key) => {
    if (key.name === "q" && state === "home") {
      backend.sendCommand({ cmd: "quit" })
      renderer.destroy()
    }
    if (key.name === "escape" && (state === "results" || state === "settings")) {
      back()
    }
  })

  const handleSaveSettings = useCallback((newConfig: Config) => {
    backend.sendCommand({ cmd: "set-config", config: newConfig })
    setConfig(newConfig)
    back()
  }, [backend, back])

  if (!backend.ready) {
    return (
      <box flexDirection="column" width="100%" height="100%">
        <box flexGrow={1} justifyContent="center" alignItems="center">
          <text fg={colors.dimmed}>Starting backend...</text>
        </box>
      </box>
    )
  }

  // HomeView manages its own full layout including StatusBar
  if (state === "home") {
    return (
      <HomeView
        eyeLogoRows={EYE_LOGO_ROWS}
        textLogoRaw={TEXT_LOGO_RAW}
        configErrors={configErrors}
        configWarnings={configWarnings}
        onStartNewRun={() => {
          setWizardMode("new")
          setActiveRunResumeMode(false)
          setActiveRunAutoStart(false)
          go("new_run_wizard")
        }}
        onResumeRun={() => {
          setWizardMode("resume")
          setActiveRunResumeMode(false)
          setActiveRunAutoStart(false)
          go("new_run_wizard")
        }}
        onSettings={() => go("settings")}
      />
    )
  }

  // NewRunWizard manages its own layout (logo + wizard)
  if (state === "new_run_wizard") {
    return (
      <NewRunWizard
        mode={wizardMode}
        eyeLogoRows={EYE_LOGO_ROWS}
        textLogoRaw={TEXT_LOGO_RAW}
        config={config!}
        sendCommand={backend.sendCommand}
        onEvent={backend.onEvent}
        onStartRun={({ inputRootDir, outputTsvPath, runMode, singleBoxDir, resumeStats }) => {
          const resumeMode = wizardMode === "resume"
          setActiveRunResumeMode(resumeMode)
          setActiveRunAutoStart(true)
          setActiveRunContext({ inputRootDir, outputTsvPath, runMode, singleBoxDir, resumeStats })
          backend.clearEvents()
          setResults([])
          go("active_run")
        }}
        onCancel={() => go("home")}
      />
    )
  }

  // ActiveRunView manages its own full layout
  if (state === "active_run") {
    return (
      <ActiveRunView
        sendCommand={backend.sendCommand}
        onEvent={backend.onEvent}
        onComplete={() => {
          setActiveRunResumeMode(false)
          setActiveRunAutoStart(false)
          go("home")
        }}
        onQuit={() => {
          setActiveRunResumeMode(false)
          setActiveRunAutoStart(false)
          go("home")
        }}
        resume={activeRunResumeMode}
        autoStart={activeRunAutoStart}
        outputTsvPath={activeRunContext.outputTsvPath || (config?.output_tsv_path ?? "")}
        inputRootDir={activeRunContext.inputRootDir || (config?.input_root_dir ?? "")}
        runMode={activeRunContext.runMode}
        singleBoxDir={activeRunContext.singleBoxDir}
        resumeStats={activeRunContext.resumeStats}
        eyeLogoRows={EYE_LOGO_ROWS}
        textLogoRaw={TEXT_LOGO_RAW}
      />
    )
  }

  // All other views use a shared shell with StatusBar
  const bar = statusBarContent[state] ?? { left: "", right: "" }

  return (
    <box flexDirection="column" width="100%" height="100%">
      <box flexGrow={1}>
        {state === "results" && (
          <ResultsView results={results} onBack={() => go("home")} />
        )}

        {state === "settings" && config && (
          <SettingsView
            config={config}
            onSave={handleSaveSettings}
            onCancel={back}
          />
        )}
      </box>

      <StatusBar left={bar.left} right={bar.right} />
    </box>
  )
}
