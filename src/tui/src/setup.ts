import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "fs"
import { homedir } from "os"
import { join } from "path"

const ODINSLIST_HOME = join(homedir(), ".odinslist")
const VENV_DIR = join(ODINSLIST_HOME, "venv")
const VENV_PYTHON = join(VENV_DIR, "bin", "python")
const BACKEND_DIR = join(ODINSLIST_HOME, "backend")
const BACKEND_VERSION_FILE = join(BACKEND_DIR, ".odinslist-backend-version")
const INSTALLED_VERSION_FILE = join(VENV_DIR, ".odinslist-backend-version")

export { VENV_PYTHON, BACKEND_DIR, ODINSLIST_HOME }

function log(message: string) {
  process.stdout.write(`  ${message}\n`)
}

async function findPython(): Promise<string | null> {
  for (const candidate of ["python3", "python"]) {
    try {
      const proc = Bun.spawn([candidate, "--version"], {
        stdout: "pipe",
        stderr: "pipe",
      })

      const stdout = await new Response(proc.stdout).text()
      const stderr = await new Response(proc.stderr).text()
      await proc.exited
      if (proc.exitCode !== 0) continue

      const output = `${stdout}\n${stderr}`.trim()
      const match = output.match(/Python (\d+)\.(\d+)/)
      if (!match) continue

      const major = parseInt(match[1], 10)
      const minor = parseInt(match[2], 10)
      if (major === 3 && minor >= 10) {
        return candidate
      }
    } catch {
      continue
    }
  }

  return null
}

async function hasWorkingPip(): Promise<boolean> {
  if (!existsSync(VENV_PYTHON)) return false

  try {
    const proc = Bun.spawn([VENV_PYTHON, "-m", "pip", "--version"], {
      stdout: "pipe",
      stderr: "pipe",
    })
    await proc.exited
    return proc.exitCode === 0
  } catch {
    return false
  }
}

function readVersion(path: string): string | null {
  if (!existsSync(path)) return null
  try {
    return readFileSync(path, "utf8").trim() || null
  } catch {
    return null
  }
}

export async function ensureBackend(): Promise<void> {
  if (process.env.ODINSLIST_PYTHON) return
  if (!existsSync(BACKEND_DIR)) return

  const backendVersion = readVersion(BACKEND_VERSION_FILE)
  const installedVersion = readVersion(INSTALLED_VERSION_FILE)
  const hasVenvPython = existsSync(VENV_PYTHON)
  const pipHealthy = await hasWorkingPip()

  const needsVenvRecreate = !hasVenvPython || !pipHealthy
  const needsReinstallForVersion = Boolean(backendVersion) && backendVersion !== installedVersion

  if (!needsVenvRecreate && !needsReinstallForVersion) return

  console.log("\n⚙ First-run setup\n")

  if (needsVenvRecreate) {
    if (existsSync(VENV_DIR)) {
      log("Detected broken virtual environment. Recreating...")
      rmSync(VENV_DIR, { recursive: true, force: true })
    }

    log("Looking for Python 3.10+...")
    const pythonBin = await findPython()
    if (!pythonBin) {
      console.error(
        "\n✗ Python 3.10+ is required but was not found.\n" +
        "  Install it from: https://www.python.org/downloads/\n",
      )
      process.exit(1)
    }
    log(`Found: ${pythonBin}`)

    mkdirSync(ODINSLIST_HOME, { recursive: true })

    log("Creating virtual environment...")
    const venvProc = Bun.spawn([pythonBin, "-m", "venv", VENV_DIR], {
      stdout: "inherit",
      stderr: "inherit",
    })
    await venvProc.exited
    if (venvProc.exitCode !== 0) {
      console.error("\n✗ Failed to create virtual environment.")
      process.exit(1)
    }
  } else if (needsReinstallForVersion) {
    log(`Backend changed (${installedVersion ?? "unknown"} -> ${backendVersion}).`)
  }

  log("Installing backend dependencies (this may take a minute)...")
  const pipProc = Bun.spawn(
    [VENV_PYTHON, "-m", "pip", "install", "--quiet", BACKEND_DIR],
    {
      stdout: "inherit",
      stderr: "inherit",
    },
  )
  await pipProc.exited
  if (pipProc.exitCode !== 0) {
    console.error(
      "\n✗ Failed to install backend dependencies.\n" +
      `  Try manually: ${VENV_PYTHON} -m pip install ${BACKEND_DIR}\n`,
    )
    process.exit(1)
  }

  if (backendVersion) {
    writeFileSync(INSTALLED_VERSION_FILE, `${backendVersion}\n`)
  }

  log("Setup complete!\n")
}
