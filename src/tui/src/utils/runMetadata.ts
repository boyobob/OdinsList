import { mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "fs"
import { homedir } from "os"
import { basename, extname, join, resolve } from "path"

export type RunMode = "batch" | "single_box"

export type RunStatus = "running" | "paused" | "cancelled" | "complete"

export interface ResumeProgress {
  processedImages: number
  processedBoxCount: number
  processedByBox: Record<string, number>
  uniqueBoxes: string[]
  completedBoxes: string[]
}

export interface TsvIdentity {
  path: string
  dev: number
  inode: number
  size: number
  mtimeNs: number
  sha256: string
}

export interface RunMetadataRecord {
  run_id: string
  status: RunStatus
  created_at: string
  updated_at: string

  run_mode: RunMode
  input_root_dir: string
  single_box_dir: string
  boxes_in_scope: string[]

  total_boxes: number
  total_images: number
  box_image_totals: Record<string, number>

  processed_images: number
  processed_box_count: number
  processed_by_box: Record<string, number>
  completed_boxes: string[]

  tsv_path: string
  tsv_dev: number
  tsv_inode: number
  tsv_size: number
  tsv_mtime_ns: number
  tsv_sha256: string
}

const METADATA_DIR = process.env.ODINSLIST_RUN_METADATA_DIR?.trim()
  ? resolve(process.env.ODINSLIST_RUN_METADATA_DIR.trim())
  : join(homedir(), ".config", "odinslist", "run_metadata")

function nowIso(): string {
  return new Date().toISOString()
}

function canonical(path: string): string {
  return resolve(path.trim())
}

function runIdFromPath(tsvPath: string): string {
  const base = basename(tsvPath, extname(tsvPath)) || "run"
  const suffix = `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`
  const normalized = base.toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "") || "run"
  return `${normalized}-${suffix}`
}

function toHex(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer)
  let result = ""
  for (const b of bytes) {
    result += b.toString(16).padStart(2, "0")
  }
  return result
}

async function sha256File(path: string): Promise<string> {
  const file = Bun.file(path)
  if (!(await file.exists())) return ""
  const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer())
  return toHex(digest)
}

function statIdentity(path: string): { dev: number; inode: number; size: number; mtimeNs: number } {
  const st = statSync(path)
  return {
    dev: Number(st.dev ?? 0),
    inode: Number((st as any).ino ?? 0),
    size: Number(st.size ?? 0),
    mtimeNs: Number((st as any).mtimeNs ?? Math.floor((st.mtimeMs ?? 0) * 1_000_000)),
  }
}

export function metadataDir(): string {
  mkdirSync(METADATA_DIR, { recursive: true })
  return METADATA_DIR
}

export function metadataPathForRun(runId: string): string {
  return join(metadataDir(), `${runId}.json`)
}

function listMetadataFiles(): string[] {
  try {
    return readdirSync(metadataDir())
      .filter((name) => name.toLowerCase().endsWith(".json"))
      .map((name) => join(metadataDir(), name))
      .sort()
  } catch {
    return []
  }
}

export function readRunMetadata(path: string): RunMetadataRecord | null {
  try {
    const raw = readFileSync(path, "utf-8")
    const parsed = JSON.parse(raw) as RunMetadataRecord
    if (!parsed || !parsed.run_id) return null
    return parsed
  } catch {
    return null
  }
}

export function loadRunMetadataById(runId: string): RunMetadataRecord | null {
  return readRunMetadata(metadataPathForRun(runId))
}

export function saveRunMetadata(record: RunMetadataRecord): void {
  const payload = {
    ...record,
    updated_at: nowIso(),
  }
  writeFileSync(metadataPathForRun(record.run_id), `${JSON.stringify(payload, null, 2)}\n`, "utf-8")
}

export async function computeTsvIdentity(tsvPath: string): Promise<TsvIdentity | null> {
  try {
    const resolved = canonical(tsvPath)
    const file = Bun.file(resolved)
    if (!(await file.exists())) return null
    const st = statIdentity(resolved)
    return {
      path: resolved,
      dev: st.dev,
      inode: st.inode,
      size: st.size,
      mtimeNs: st.mtimeNs,
      sha256: await sha256File(resolved),
    }
  } catch {
    return null
  }
}

function isSamePath(a: string, b: string): boolean {
  return canonical(a) === canonical(b)
}

function isSameInode(a: TsvIdentity, b: RunMetadataRecord): boolean {
  return a.dev > 0 && a.inode > 0 && a.dev === b.tsv_dev && a.inode === b.tsv_inode
}

function isSameHash(a: TsvIdentity, b: RunMetadataRecord): boolean {
  return !!a.sha256 && !!b.tsv_sha256 && a.sha256 === b.tsv_sha256
}

export interface MetadataLookupResult {
  record: RunMetadataRecord
  matchedBy: "path" | "inode" | "hash"
}

export async function findRunMetadataForTsv(tsvPath: string): Promise<MetadataLookupResult | null> {
  const resolvedTsv = canonical(tsvPath)
  const files = listMetadataFiles()

  for (const file of files) {
    const record = readRunMetadata(file)
    if (!record) continue
    if (record.tsv_path && isSamePath(record.tsv_path, resolvedTsv)) {
      return { record, matchedBy: "path" }
    }
  }

  const identity = await computeTsvIdentity(resolvedTsv)
  if (!identity) return null

  for (const file of files) {
    const record = readRunMetadata(file)
    if (!record) continue
    if (isSameInode(identity, record)) {
      return { record, matchedBy: "inode" }
    }
  }

  for (const file of files) {
    const record = readRunMetadata(file)
    if (!record) continue
    if (isSameHash(identity, record)) {
      return { record, matchedBy: "hash" }
    }
  }

  return null
}

export async function bindMetadataToTsv(record: RunMetadataRecord, tsvPath: string): Promise<RunMetadataRecord> {
  const identity = await computeTsvIdentity(tsvPath)
  const resolved = canonical(tsvPath)
  if (!identity) {
    return {
      ...record,
      tsv_path: resolved,
      tsv_dev: 0,
      tsv_inode: 0,
      tsv_size: 0,
      tsv_mtime_ns: 0,
      tsv_sha256: "",
      updated_at: nowIso(),
    }
  }

  return {
    ...record,
    tsv_path: identity.path,
    tsv_dev: identity.dev,
    tsv_inode: identity.inode,
    tsv_size: identity.size,
    tsv_mtime_ns: identity.mtimeNs,
    tsv_sha256: identity.sha256,
    updated_at: nowIso(),
  }
}

export async function makeRunMetadata(args: {
  runId?: string
  status?: RunStatus
  runMode: RunMode
  inputRootDir: string
  singleBoxDir: string
  boxesInScope: string[]
  totalBoxes: number
  totalImages: number
  boxImageTotals: Record<string, number>
  progress: ResumeProgress
  tsvPath: string
}): Promise<RunMetadataRecord> {
  const created = nowIso()
  const base: RunMetadataRecord = {
    run_id: args.runId || runIdFromPath(args.tsvPath),
    status: args.status || "running",
    created_at: created,
    updated_at: created,

    run_mode: args.runMode,
    input_root_dir: canonical(args.inputRootDir),
    single_box_dir: args.singleBoxDir ? canonical(args.singleBoxDir) : "",
    boxes_in_scope: [...args.boxesInScope],

    total_boxes: args.totalBoxes,
    total_images: args.totalImages,
    box_image_totals: { ...args.boxImageTotals },

    processed_images: args.progress.processedImages,
    processed_box_count: args.progress.processedBoxCount,
    processed_by_box: { ...args.progress.processedByBox },
    completed_boxes: [...args.progress.completedBoxes],

    tsv_path: canonical(args.tsvPath),
    tsv_dev: 0,
    tsv_inode: 0,
    tsv_size: 0,
    tsv_mtime_ns: 0,
    tsv_sha256: "",
  }

  return bindMetadataToTsv(base, args.tsvPath)
}

export function withProgress(record: RunMetadataRecord, progress: ResumeProgress): RunMetadataRecord {
  return {
    ...record,
    processed_images: progress.processedImages,
    processed_box_count: progress.processedBoxCount,
    processed_by_box: { ...progress.processedByBox },
    completed_boxes: [...progress.completedBoxes],
    updated_at: nowIso(),
  }
}

export function withStatus(record: RunMetadataRecord, status: RunStatus): RunMetadataRecord {
  return {
    ...record,
    status,
    updated_at: nowIso(),
  }
}
