import { afterEach, beforeEach, describe, expect, test } from "bun:test"
import { mkdirSync, rmSync } from "fs"
import { join, resolve } from "path"
import type { Config } from "../types"
import { analyzeResumeTsv } from "./resumeRun"
import { makeRunMetadata, saveRunMetadata } from "./runMetadata"

const TMP_ROOT = "/tmp/odinslist-resume-run-tests"
const METADATA_ROOT = join(TMP_ROOT, "metadata")

function baseConfig(): Config {
  return {
    input_root_dir: "",
    output_tsv_path: "",
    gcd_db_path: "",
    vlm_base_url: "http://127.0.0.1:8000/v1",
    vlm_model: "test-model",
    comicvine_api_key: "",
    gcd_enabled: true,
    comicvine_enabled: true,
    run_mode: "batch",
    single_box_dir: "",
  }
}

beforeEach(() => {
  process.env.ODINSLIST_RUN_METADATA_DIR = METADATA_ROOT
})

afterEach(() => {
  delete process.env.ODINSLIST_RUN_METADATA_DIR
  rmSync(TMP_ROOT, { recursive: true, force: true })
})

describe("analyzeResumeTsv", () => {
  test("uses the configured run scope when the selected TSV matches config", async () => {
    const root = join(TMP_ROOT, "batch-root")
    mkdirSync(join(root, "Box_01"), { recursive: true })
    mkdirSync(join(root, "Box_02"), { recursive: true })

    const tsvPath = join(root, "All_Boxes.tsv")
    await Bun.write(
      tsvPath,
      [
        "title\tissue_number\tmonth\tyear\tpublisher\tbox\tfilename\tnotes\tconfidence",
        "A\t1\tJAN\t1990\tMarvel\tBox_01\t001.jpg\t\thigh",
        "B\t2\tFEB\t1991\tMarvel\tBox_02\t002.jpg\t\tmedium",
      ].join("\n"),
    )

    const config = baseConfig()
    config.input_root_dir = root
    config.output_tsv_path = tsvPath
    config.run_mode = "batch"

    const analysis = await analyzeResumeTsv(tsvPath, config)

    expect(analysis.runMode).toBe("batch")
    expect(analysis.inputRootDir).toBe(resolve(root))
    expect(analysis.singleBoxDir).toBe("")
    expect(analysis.processedImages).toBe(2)
    expect(analysis.processedBoxCount).toBe(2)
  })

  test("infers a single-box resume from the default box-local TSV path", async () => {
    const root = join(TMP_ROOT, "single-root")
    const boxDir = join(root, "Box_16")
    mkdirSync(boxDir, { recursive: true })

    const tsvPath = join(boxDir, "Box_16.tsv")
    await Bun.write(
      tsvPath,
      [
        "title\tissue_number\tmonth\tyear\tpublisher\tbox\tfilename\tnotes\tconfidence",
        "A\t1\tJAN\t1990\tMarvel\tBox_16\t001.jpg\t\thigh",
        "B\t2\tFEB\t1991\tMarvel\tBox_16\t002.jpg\t\tmedium",
      ].join("\n"),
    )

    const analysis = await analyzeResumeTsv(tsvPath, baseConfig())

    expect(analysis.runMode).toBe("single_box")
    expect(analysis.inputRootDir).toBe(resolve(root))
    expect(analysis.singleBoxDir).toBe(resolve(boxDir))
    expect(analysis.processedImages).toBe(2)
    expect(analysis.uniqueBoxes).toEqual(["Box_16"])
  })

  test("infers a batch resume from a root-level All_Boxes.tsv file", async () => {
    const root = join(TMP_ROOT, "batch-root-heuristic")
    mkdirSync(join(root, "Box_01"), { recursive: true })
    mkdirSync(join(root, "Box_02"), { recursive: true })
    mkdirSync(join(root, "Box_03"), { recursive: true })

    const tsvPath = join(root, "All_Boxes.tsv")
    await Bun.write(
      tsvPath,
      [
        "title\tissue_number\tmonth\tyear\tpublisher\tbox\tfilename\tnotes\tconfidence",
        "A\t1\tJAN\t1990\tMarvel\tBox_01\t001.jpg\t\thigh",
        "B\t2\tFEB\t1991\tMarvel\tBox_02\t002.jpg\t\tmedium",
        "C\t3\tMAR\t1992\tMarvel\tBox_02\t003.jpg\t\tmedium",
      ].join("\n"),
    )

    const analysis = await analyzeResumeTsv(tsvPath, baseConfig())

    expect(analysis.runMode).toBe("batch")
    expect(analysis.inputRootDir).toBe(resolve(root))
    expect(analysis.processedImages).toBe(3)
    expect(analysis.processedByBox).toEqual({
      Box_01: 1,
      Box_02: 2,
    })
  })

  test("prefers stored run metadata scope over config when metadata matches the TSV", async () => {
    const root = join(TMP_ROOT, "metadata-match-root")
    const singleBoxDir = join(root, "Box_07")
    mkdirSync(singleBoxDir, { recursive: true })

    const tsvPath = join(singleBoxDir, "custom_resume.tsv")
    await Bun.write(
      tsvPath,
      [
        "title\tissue_number\tmonth\tyear\tpublisher\tbox\tfilename\tnotes\tconfidence",
        "A\t1\tJAN\t1990\tMarvel\tBox_07\t001.jpg\t\thigh",
      ].join("\n"),
    )

    const metadata = await makeRunMetadata({
      status: "paused",
      runMode: "single_box",
      inputRootDir: root,
      singleBoxDir,
      boxesInScope: ["Box_07"],
      totalBoxes: 1,
      totalImages: 10,
      boxImageTotals: { Box_07: 10 },
      progress: {
        processedImages: 1,
        processedBoxCount: 1,
        processedByBox: { Box_07: 1 },
        uniqueBoxes: ["Box_07"],
        completedBoxes: [],
      },
      tsvPath,
    })
    saveRunMetadata(metadata)

    const config = baseConfig()
    config.input_root_dir = "/wrong/root"
    config.run_mode = "batch"
    config.output_tsv_path = "/wrong/path.tsv"

    const analysis = await analyzeResumeTsv(tsvPath, config)
    expect(analysis.runId).toBe(metadata.run_id)
    expect(analysis.metadataMatchedBy).toBe("path")
    expect(analysis.runMode).toBe("single_box")
    expect(analysis.inputRootDir).toBe(resolve(root))
    expect(analysis.singleBoxDir).toBe(resolve(singleBoxDir))
  })
})
