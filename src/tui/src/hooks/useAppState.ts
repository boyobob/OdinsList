import { useState, useCallback } from "react"
import type { AppState } from "../types"

interface AppStateManager {
  state: AppState
  previousState: AppState | null
  go: (next: AppState) => void
  back: () => void
}

export function useAppState(initial: AppState): AppStateManager {
  const [state, setState] = useState<AppState>(initial)
  const [previousState, setPreviousState] = useState<AppState | null>(null)

  const go = useCallback((next: AppState) => {
    setState(current => {
      setPreviousState(current)
      return next
    })
  }, [])

  const back = useCallback(() => {
    setPreviousState(null)
    setState(previousState ?? "home")
  }, [previousState])

  return { state, previousState, go, back }
}
