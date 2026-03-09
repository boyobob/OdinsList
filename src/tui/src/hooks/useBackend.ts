import { useState, useEffect, useCallback, useRef } from "react"
import type { Command, ScanEvent } from "../types"
import { resolve } from "path"
import { existsSync } from "fs"
import { BACKEND_DIR, VENV_PYTHON } from "../setup"

interface BackendState {
  ready: boolean
  events: ScanEvent[]
}

export function useBackend() {
  const [state, setState] = useState<BackendState>({ ready: false, events: [] })
  const procRef = useRef<ReturnType<typeof Bun.spawn> | null>(null)
  const listenersRef = useRef<Set<(event: ScanEvent) => void>>(new Set())

  useEffect(() => {
    const devPython = process.env.ODINSLIST_PYTHON
    const hasInstalledBackend = existsSync(BACKEND_DIR)
    const backendRoot = process.env.ODINSLIST_ROOT
      || (hasInstalledBackend ? BACKEND_DIR : resolve(import.meta.dir, "../../../"))
    const pythonBin = devPython || (existsSync(VENV_PYTHON) ? VENV_PYTHON : "python3")
    const env: Record<string, string> = { ...process.env } as Record<string, string>

    const usingSourceBackend = Boolean(process.env.ODINSLIST_ROOT) || !hasInstalledBackend
    if (usingSourceBackend || devPython) {
      const currentPythonPath = process.env.PYTHONPATH || ""
      env.PYTHONPATH = currentPythonPath
        ? `${backendRoot}:${currentPythonPath}`
        : backendRoot
    }

    const proc = Bun.spawn([pythonBin, "-m", "odinslist", "--ipc"], {
      stdin: "pipe",
      stdout: "pipe",
      stderr: "pipe",
      cwd: backendRoot,
      env,
    })

    procRef.current = proc

    // Read stdout line by line
    async function readLines() {
      const reader = proc.stdout.getReader()
      const decoder = new TextDecoder()
      let buffer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() || ""

        for (const line of lines) {
          if (!line.trim()) continue
          try {
            const event = JSON.parse(line) as ScanEvent
            if (event.event === "ready") {
              setState(s => ({ ...s, ready: true }))
            }
            setState(s => ({ ...s, events: [...s.events, event] }))
            for (const listener of listenersRef.current) {
              listener(event)
            }
          } catch {
            // Skip malformed lines
          }
        }
      }
    }

    readLines()

    return () => {
      const stdin = proc.stdin as import("bun").FileSink
      stdin.write(JSON.stringify({ cmd: "quit" }) + "\n")
      stdin.flush()
      proc.kill()
      procRef.current = null
    }
  }, [])

  const sendCommand = useCallback((command: Command) => {
    const proc = procRef.current
    if (!proc) return
    const stdin = proc.stdin as import("bun").FileSink
    stdin.write(JSON.stringify(command) + "\n")
    stdin.flush()
  }, [])

  const onEvent = useCallback((listener: (event: ScanEvent) => void) => {
    listenersRef.current.add(listener)
    return () => { listenersRef.current.delete(listener) }
  }, [])

  const clearEvents = useCallback(() => {
    setState(s => ({ ...s, events: [] }))
  }, [])

  return {
    ready: state.ready,
    events: state.events,
    sendCommand,
    onEvent,
    clearEvents,
  }
}
