// App states
export type AppState = "setup" | "home" | "new_run_wizard" | "active_run" | "results" | "settings"
export type RunMode = "batch" | "single_box"

// IPC Commands (TUI -> Python)
export type Command =
  | { cmd: "scan"; resume: boolean }
  | { cmd: "scan-preview" }
  | { cmd: "pause" }
  | { cmd: "resume" }
  | { cmd: "cancel" }
  | { cmd: "get-config" }
  | { cmd: "set-config"; config: Partial<Config> }
  | { cmd: "validate-config" }
  | { cmd: "list-dirs"; path: string }
  | { cmd: "quit" }

// IPC Events (Python -> TUI)
export type ScanEvent =
  | { event: "ready"; backend_cwd?: string; backend_module?: string }
  | { event: "scan_started"; images_dir: string; output_path: string; resume: boolean }
  | { event: "boxes"; boxes: BoxInfo[] }
  | { event: "config"; config: Config & { is_first_run: boolean } }
  | { event: "scan_preview"; total_images: number; boxes: BoxInfo[] }
  | { event: "BoxStarted"; box_name: string; image_count: number }
  | { event: "BoxFinished"; box_name: string; results_count: number }
  | { event: "ImageLoading"; filename: string; image_path: string }
  | { event: "VLMExtracting" }
  | { event: "VLMResult"; title: string; issue: string; publisher: string; year: string }
  | { event: "GCDSearching"; strategy: number; title: string; issue: string }
  | { event: "GCDMatchFound"; title: string; confidence: number }
  | { event: "GCDNoMatch"; strategy: number }
  | { event: "ComicVineSearching"; stage: string }
  | { event: "VisualComparing"; title: string; issue: string }
  | { event: "ComicVineMatchFound"; title: string; confidence: number }
  | { event: "ScanComplete"; result: ComicResult; confidence: string }
  | { event: "ScanError"; filename: string; error: string }
  | { event: "run_complete" }
  | { event: "cancelled" }
  | { event: "paused" }
  | { event: "resumed" }
  | { event: "error"; command: string; message: string }
  | { event: "config_validation"; errors: string[]; warnings: string[] }
  | { event: "dirs"; path: string; dirs: string[]; error?: string }

export interface BoxInfo {
  name: string
  count: number
}

export interface ComicResult {
  title: string
  issue_number: string
  month: string
  year: string
  publisher: string
  box: string
  filename: string
  notes: string
  confidence: string
}

export interface Config {
  input_root_dir: string
  output_tsv_path: string
  gcd_db_path: string
  vlm_base_url: string
  vlm_model: string
  comicvine_api_key: string
  gcd_enabled: boolean
  comicvine_enabled: boolean
  run_mode: RunMode
  single_box_dir: string
}

export interface ResumeStats {
  runId: string | null
  processedImages: number
  processedBoxCount: number
  processedByBox: Record<string, number>
  uniqueBoxes: string[]
  completedBoxes: string[]
  metadataMatchedBy: "path" | "inode" | "hash" | null
}

// Log entry for the event log
export interface LogEntry {
  tag: string
  message: string
  color: string
}
