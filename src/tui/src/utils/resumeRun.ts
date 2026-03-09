import { existsSync, statSync } from "fs"
import { basename, dirname, extname, join, resolve } from "path"
import type { Config } from "../types"
import {
  bindMetadataToTsv,
  findRunMetadataForTsv,
  saveRunMetadata,
  withProgress,
  type RunMode,
} from "./runMetadata"

export interface ResumeRunAnalysis {
  runId: string | null
  inputRootDir: string
  runMode: RunMode
  singleBoxDir: string
  outputTsvPath: string
  processedImages: number
  processedBoxCount: number
  processedByBox: Record<string, number>
  uniqueBoxes: string[]
  completedBoxes: string[]
  metadataMatchedBy: "path" | "inode" | "hash" | null
}

function fieldIndexMap(header: string[]): Record<string, number> {
  const map: Record<string, number> = {}
  for (let i = 0; i < header.length; i += 1) {
    map[header[i]] = i
  }
  return map
}

function dirExists(path: string): boolean {
  try {
    return existsSync(path) && statSync(path).isDirectory()
  } catch {
    return false
  }
}

function uniquePaths(paths: string[]): string[] {
  const seen = new Set<string>()
  const result: string[] = []
  for (const path of paths) {
    if (!path) continue
    const resolved = resolve(path)
    if (seen.has(resolved)) continue
    seen.add(resolved)
    result.push(resolved)
  }
  return result
}

function findBatchRoot(boxes: string[], candidates: string[]): string | null {
  for (const candidate of uniquePaths(candidates)) {
    if (!dirExists(candidate)) continue
    const hasAllBoxes = boxes.every((box) => dirExists(join(candidate, box)))
    if (hasAllBoxes) return candidate
  }
  return null
}

function inferRunScope(
  outputTsvPath: string,
  uniqueBoxes: string[],
  config: Config,
): Pick<ResumeRunAnalysis, "inputRootDir" | "runMode" | "singleBoxDir" | "outputTsvPath"> {
  const resolvedTsvPath = resolve(outputTsvPath)
  const selectedName = basename(resolvedTsvPath).toLowerCase()
  const selectedNameNoExt = basename(resolvedTsvPath, extname(resolvedTsvPath))
  const parentDir = dirname(resolvedTsvPath)
  const parentName = basename(parentDir)
  const grandparentDir = dirname(parentDir)
  const resolvedConfigTsvPath = config.output_tsv_path.trim()
    ? resolve(config.output_tsv_path.trim())
    : ""
  const configRootDir = config.input_root_dir.trim()
    ? resolve(config.input_root_dir.trim())
    : ""
  const configRunMode: RunMode = config.run_mode === "single_box" ? "single_box" : "batch"
  const resolvedConfigSingleBoxDir = config.single_box_dir.trim()
    ? resolve(config.single_box_dir.trim())
    : ""
  const firstBox = uniqueBoxes[0] ?? ""

  if (resolvedConfigTsvPath && resolvedTsvPath === resolvedConfigTsvPath && configRootDir) {
    return {
      inputRootDir: configRootDir,
      runMode: configRunMode,
      singleBoxDir: configRunMode === "single_box"
        ? resolvedConfigSingleBoxDir || join(configRootDir, firstBox)
        : "",
      outputTsvPath: resolvedTsvPath,
    }
  }

  if (selectedName === "all_boxes.tsv") {
    return {
      inputRootDir: parentDir,
      runMode: "batch",
      singleBoxDir: "",
      outputTsvPath: resolvedTsvPath,
    }
  }

  if (selectedNameNoExt === parentName) {
    return {
      inputRootDir: grandparentDir,
      runMode: "single_box",
      singleBoxDir: parentDir,
      outputTsvPath: resolvedTsvPath,
    }
  }

  if (uniqueBoxes.length > 1) {
    const batchRoot = findBatchRoot(uniqueBoxes, [configRootDir, parentDir, grandparentDir])
    if (!batchRoot) {
      throw new Error("Could not infer the batch root directory for this TSV")
    }
    return {
      inputRootDir: batchRoot,
      runMode: "batch",
      singleBoxDir: "",
      outputTsvPath: resolvedTsvPath,
    }
  }

  if (configRootDir && firstBox && dirExists(join(configRootDir, firstBox))) {
    if (configRunMode === "batch") {
      return {
        inputRootDir: configRootDir,
        runMode: "batch",
        singleBoxDir: "",
        outputTsvPath: resolvedTsvPath,
      }
    }
    return {
      inputRootDir: configRootDir,
      runMode: "single_box",
      singleBoxDir: resolvedConfigSingleBoxDir || join(configRootDir, firstBox),
      outputTsvPath: resolvedTsvPath,
    }
  }

  const fallbackBatchRoot = findBatchRoot(uniqueBoxes, [parentDir, grandparentDir])
  if (fallbackBatchRoot) {
    return {
      inputRootDir: fallbackBatchRoot,
      runMode: "batch",
      singleBoxDir: "",
      outputTsvPath: resolvedTsvPath,
    }
  }

  if (firstBox && parentName === firstBox) {
    return {
      inputRootDir: grandparentDir,
      runMode: "single_box",
      singleBoxDir: parentDir,
      outputTsvPath: resolvedTsvPath,
    }
  }

  throw new Error("Could not infer whether this TSV belongs to a batch or single-box run")
}

export async function analyzeResumeTsv(tsvPath: string, config: Config): Promise<ResumeRunAnalysis> {
  const trimmedPath = tsvPath.trim()
  if (!trimmedPath) {
    throw new Error("Resume TSV path is required")
  }

  const resolvedPath = resolve(trimmedPath)
  if (extname(resolvedPath).toLowerCase() !== ".tsv") {
    throw new Error("Resume file must be a .tsv")
  }

  const file = Bun.file(resolvedPath)
  if (!(await file.exists())) {
    throw new Error(`Resume TSV not found at ${resolvedPath}`)
  }

  const raw = await file.text()
  const lines = raw.split(/\r?\n/).filter((line) => line.trim().length > 0)
  if (lines.length <= 1) {
    throw new Error("Resume TSV has no processed rows")
  }

  const header = lines[0].split("\t").map((field) => field.trim())
  const idx = fieldIndexMap(header)
  if (idx.box === undefined || idx.filename === undefined) {
    throw new Error("Resume TSV is missing required columns")
  }

  const processedByBox: Record<string, number> = {}

  for (let i = 1; i < lines.length; i += 1) {
    const fields = lines[i].split("\t")
    const box = (fields[idx.box] ?? "").trim()
    const filename = (fields[idx.filename] ?? "").trim()
    if (!box || !filename) continue
    processedByBox[box] = (processedByBox[box] ?? 0) + 1
  }

  const uniqueBoxes = Object.keys(processedByBox).sort()
  const processedImages = uniqueBoxes.reduce(
    (sum, box) => sum + (processedByBox[box] ?? 0),
    0,
  )

  if (processedImages === 0) {
    throw new Error("Resume TSV has no processed image rows")
  }

  const metadataHit = await findRunMetadataForTsv(resolvedPath)
  let scope: Pick<ResumeRunAnalysis, "inputRootDir" | "runMode" | "singleBoxDir" | "outputTsvPath">
  let runId: string | null = null
  let metadataMatchedBy: "path" | "inode" | "hash" | null = null

  if (metadataHit) {
    const record = metadataHit.record
    runId = record.run_id
    metadataMatchedBy = metadataHit.matchedBy
    scope = {
      inputRootDir: resolve(record.input_root_dir),
      runMode: record.run_mode === "single_box" ? "single_box" : "batch",
      singleBoxDir: record.single_box_dir ? resolve(record.single_box_dir) : "",
      outputTsvPath: resolvedPath,
    }

    const completedBoxes = Object.keys(processedByBox)
      .filter((box) => (processedByBox[box] ?? 0) >= 1)
      .sort()
    const refreshed = withProgress(record, {
      processedImages,
      processedBoxCount: uniqueBoxes.length,
      processedByBox,
      uniqueBoxes,
      completedBoxes,
    })
    const rebound = await bindMetadataToTsv(refreshed, resolvedPath)
    saveRunMetadata(rebound)
  } else {
    scope = inferRunScope(resolvedPath, uniqueBoxes, config)
  }

  const completedBoxes = Object.keys(processedByBox)
    .filter((box) => (processedByBox[box] ?? 0) > 0)
    .sort()

  return {
    runId,
    ...scope,
    processedImages,
    processedBoxCount: uniqueBoxes.length,
    processedByBox,
    uniqueBoxes,
    completedBoxes,
    metadataMatchedBy,
  }
}
