import type { HalfBlockArt, HalfBlockCell } from "./pixelArt";

// ── Chafa detection (cached) ────────────────────────────────────────────────

let chafaAvailable: boolean | null = null;

export async function detectChafa(): Promise<boolean> {
  const disableChafa = (process.env.ODINSLIST_DISABLE_CHAFA ?? "").trim().toLowerCase();
  if (disableChafa === "1" || disableChafa === "true" || disableChafa === "yes") {
    chafaAvailable = false;
    return false;
  }

  if (chafaAvailable !== null) return chafaAvailable;

  try {
    const proc = Bun.spawn(["which", "chafa"], {
      stdout: "pipe",
      stderr: "pipe",
    });
    const code = await proc.exited;
    chafaAvailable = code === 0;
  } catch {
    chafaAvailable = false;
  }

  return chafaAvailable;
}

// ── ANSI SGR parser ─────────────────────────────────────────────────────────

interface CellStyle {
  fg: string | null;
  bg: string | null;
}

/**
 * Parse chafa `--format=symbols` output into HalfBlockCell[][].
 *
 * Handles SGR color sequences and ignores other CSI sequences such as
 * cursor visibility toggles, which chafa can emit even in symbols mode.
 */
export function parseAnsiArt(output: string): HalfBlockCell[][] {
  const lines = output.split("\n");
  const rows: HalfBlockCell[][] = [];

  // Strip any trailing empty lines
  while (lines.length > 0 && lines[lines.length - 1].trim() === "") {
    lines.pop();
  }

  const csiPattern = /\x1b\[([0-?]*)([ -/]*)([@-~])/g;

  for (const line of lines) {
    const cells: HalfBlockCell[] = [];
    const style: CellStyle = { fg: null, bg: null };

    let lastIndex = 0;
    let match: RegExpExecArray | null;

    csiPattern.lastIndex = 0;

    while ((match = csiPattern.exec(line)) !== null) {
      // Any visible text between the previous control sequence and this one
      const textBefore = line.slice(lastIndex, match.index);
      if (textBefore.length > 0) {
        pushChars(cells, textBefore, style);
      }

      if (match[3] === "m") {
        applySgr(style, match[1]);
      }

      lastIndex = csiPattern.lastIndex;
    }

    // Remaining text after last SGR
    const tail = line.slice(lastIndex);
    if (tail.length > 0) {
      pushChars(cells, tail, style);
    }

    if (cells.length > 0) {
      rows.push(cells);
    }
  }

  return rows;
}

/** Push each character from `text` as a cell with current style. */
function pushChars(
  cells: HalfBlockCell[],
  text: string,
  style: CellStyle,
): void {
  for (const char of text) {
    cells.push({ char, fg: style.fg, bg: style.bg });
  }
}

/** Apply a single SGR parameter string (e.g. "38;2;100;200;50" or "0"). */
function applySgr(style: CellStyle, params: string): void {
  if (params === "" || params === "0") {
    style.fg = null;
    style.bg = null;
    return;
  }

  const parts = params.split(";").map(Number);
  let i = 0;
  while (i < parts.length) {
    if (parts[i] === 38 && parts[i + 1] === 2 && i + 4 < parts.length) {
      // Truecolor foreground: 38;2;R;G;B
      style.fg = rgbToHex(parts[i + 2], parts[i + 3], parts[i + 4]);
      i += 5;
    } else if (parts[i] === 48 && parts[i + 1] === 2 && i + 4 < parts.length) {
      // Truecolor background: 48;2;R;G;B
      style.bg = rgbToHex(parts[i + 2], parts[i + 3], parts[i + 4]);
      i += 5;
    } else if (parts[i] === 0) {
      style.fg = null;
      style.bg = null;
      i += 1;
    } else {
      // Skip unrecognized codes
      i += 1;
    }
  }
}

function rgbToHex(r: number, g: number, b: number): string {
  const hex = (n: number) => Math.max(0, Math.min(255, n)).toString(16).padStart(2, "0");
  return `#${hex(r)}${hex(g)}${hex(b)}`;
}

// ── Chafa rendering ─────────────────────────────────────────────────────────

interface ChafaOptions {
  width: number;
  height: number;
}

export async function renderWithChafa(
  filePath: string,
  options: ChafaOptions,
): Promise<HalfBlockArt> {
  const proc = Bun.spawn(
    [
      "chafa",
      "--format=symbols",
      "--color-space=rgb",
      "--color-extractor=median",
      "--scale=max",
      "--optimize=0",
      "--polite=on",
      "--relative=off",
      `--size=${options.width}x${options.height}`,
      "--animate=off",
      "--bg=0a0a0f",
      filePath,
    ],
    { stdout: "pipe", stderr: "pipe" },
  );

  const [exitCode, stdout, stderr] = await Promise.all([
    proc.exited,
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);

  if (exitCode !== 0) {
    throw new Error(`chafa exited with code ${exitCode}: ${stderr}`);
  }

  const rows = parseAnsiArt(stdout);

  return {
    rows,
    width: rows[0]?.length ?? 0,
    height: rows.length,
  };
}
