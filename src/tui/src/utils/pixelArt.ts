import { PNG } from "pngjs";
import { detectChafa, renderWithChafa } from "./chafa";
import { existsSync } from "fs";
import { VENV_PYTHON } from "../setup";

// ── Types ──────────────────────────────────────────────────────────────────────

export interface HalfBlockCell {
  char: string;
  fg: string | null;
  bg: string | null;
}

export type HalfBlockArt = {
  rows: HalfBlockCell[][];
  width: number;
  height: number; // in terminal rows
};

interface ConvertOptions {
  maxWidth?: number;
  maxHeight?: number; // in terminal rows (each = 2 pixel rows)
  alphaThreshold?: number; // alpha below this = transparent (0-255, default 128)
}

// ── Constants ──────────────────────────────────────────────────────────────────

const UPPER_HALF = "\u2580"; // ▀
const LOWER_HALF = "\u2584"; // ▄
const FULL_BLOCK = "\u2588"; // █
const SPACE = " ";

// ── Helpers ────────────────────────────────────────────────────────────────────

function rgbHex(r: number, g: number, b: number): string {
  const hex = (n: number) => Math.max(0, Math.min(255, n)).toString(16).padStart(2, "0");
  return `#${hex(r)}${hex(g)}${hex(b)}`;
}

// ── Resize (nearest-neighbour) ─────────────────────────────────────────────────

interface RawImage {
  data: Buffer;
  width: number;
  height: number;
}

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

function nearestNeighbourResize(
  src: RawImage,
  dstWidth: number,
  dstHeight: number,
): RawImage {
  const dst = Buffer.alloc(dstWidth * dstHeight * 4);
  const xRatio = src.width / dstWidth;
  const yRatio = src.height / dstHeight;

  for (let y = 0; y < dstHeight; y++) {
    const srcY = Math.floor(y * yRatio);
    for (let x = 0; x < dstWidth; x++) {
      const srcX = Math.floor(x * xRatio);
      const srcIdx = (srcY * src.width + srcX) * 4;
      const dstIdx = (y * dstWidth + x) * 4;
      dst[dstIdx] = src.data[srcIdx];
      dst[dstIdx + 1] = src.data[srcIdx + 1];
      dst[dstIdx + 2] = src.data[srcIdx + 2];
      dst[dstIdx + 3] = src.data[srcIdx + 3];
    }
  }

  return { data: dst, width: dstWidth, height: dstHeight };
}

// ── Core converter ─────────────────────────────────────────────────────────────

export function convertPngToHalfBlock(
  buffer: Buffer,
  options?: ConvertOptions,
): HalfBlockArt {
  const png = PNG.sync.read(buffer);

  let img: RawImage = {
    data: png.data as unknown as Buffer,
    width: png.width,
    height: png.height,
  };

  const maxWidth = options?.maxWidth;
  const maxHeight = options?.maxHeight;
  const alphaThreshold = options?.alphaThreshold ?? 128;

  // ── Compute target dimensions ──────────────────────────────────────────────
  let targetWidth = img.width;
  let targetHeight = img.height;

  if (maxWidth !== undefined && targetWidth > maxWidth) {
    const scale = maxWidth / targetWidth;
    targetWidth = maxWidth;
    targetHeight = Math.round(targetHeight * scale);
  }

  if (maxHeight !== undefined && targetHeight > maxHeight * 2) {
    const scale = (maxHeight * 2) / targetHeight;
    targetHeight = maxHeight * 2;
    targetWidth = Math.round(targetWidth * scale);
  }

  // Ensure even pixel height for half-block pairing
  if (targetHeight % 2 !== 0) {
    targetHeight += 1;
  }

  // Resize if dimensions changed
  if (targetWidth !== img.width || targetHeight !== img.height) {
    img = nearestNeighbourResize(img, targetWidth, targetHeight);
  }

  // ── Convert pixel pairs into half-block cells ──────────────────────────────
  const termRows = img.height / 2;
  const rows: HalfBlockCell[][] = [];

  for (let row = 0; row < termRows; row++) {
    const cells: HalfBlockCell[] = [];

    for (let col = 0; col < img.width; col++) {
      // Top pixel (row * 2)
      const topIdx = (row * 2 * img.width + col) * 4;
      const topR = img.data[topIdx];
      const topG = img.data[topIdx + 1];
      const topB = img.data[topIdx + 2];
      const topA = img.data[topIdx + 3];
      const topTransparent = topA < alphaThreshold;

      // Bottom pixel (row * 2 + 1)
      const botIdx = ((row * 2 + 1) * img.width + col) * 4;
      const botR = img.data[botIdx];
      const botG = img.data[botIdx + 1];
      const botB = img.data[botIdx + 2];
      const botA = img.data[botIdx + 3];
      const botTransparent = botA < alphaThreshold;
      const topColor = rgbHex(topR, topG, topB);
      const bottomColor = rgbHex(botR, botG, botB);

      if (topTransparent && botTransparent) {
        cells.push({ char: SPACE, fg: null, bg: null });
      } else if (topTransparent) {
        cells.push({ char: LOWER_HALF, fg: bottomColor, bg: null });
      } else if (botTransparent) {
        cells.push({ char: UPPER_HALF, fg: topColor, bg: null });
      } else {
        if (topColor === bottomColor) {
          cells.push({ char: FULL_BLOCK, fg: topColor, bg: null });
        } else {
          cells.push({
            char: UPPER_HALF,
            fg: topColor,
            bg: bottomColor,
          });
        }
      }
    }

    rows.push(cells);
  }

  return {
    rows,
    width: img.width,
    height: termRows,
  };
}

// ── Async convenience loader ───────────────────────────────────────────────────

function isPngBuffer(buffer: Buffer): boolean {
  return buffer.length >= PNG_SIGNATURE.length
    && buffer.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE);
}

async function decodeImageWithPython(filePath: string): Promise<Buffer> {
  const pythonBin = existsSync(VENV_PYTHON) ? VENV_PYTHON : "python3";
  const script = [
    "import io, sys",
    "from PIL import Image, ImageOps",
    "img = Image.open(sys.argv[1])",
    "img = ImageOps.exif_transpose(img)",
    "img = img.convert('RGBA')",
    "buf = io.BytesIO()",
    "img.save(buf, format='PNG')",
    "sys.stdout.buffer.write(buf.getvalue())",
  ].join("; ");

  const proc = Bun.spawn([pythonBin, "-c", script, filePath], {
    stdout: "pipe",
    stderr: "pipe",
  });

  const [exitCode, stdout, stderr] = await Promise.all([
    proc.exited,
    new Response(proc.stdout).arrayBuffer(),
    new Response(proc.stderr).text(),
  ]);

  if (exitCode !== 0) {
    const details = stderr.trim() || "unknown image decode failure";
    throw new Error(`preview decode failed: ${details}`);
  }

  return Buffer.from(stdout);
}

export async function loadArtAsync(
  filePath: string,
  options?: ConvertOptions,
): Promise<HalfBlockArt> {
  const file = Bun.file(filePath);
  const arrayBuffer = await file.arrayBuffer();
  const sourceBuffer = Buffer.from(arrayBuffer);
  const buffer = isPngBuffer(sourceBuffer)
    ? sourceBuffer
    : await decodeImageWithPython(filePath);

  return convertPngToHalfBlock(buffer, options);
}

// ── Smart loader (chafa with fallback) ──────────────────────────────────────

export async function loadArtSmart(
  filePath: string,
  options?: ConvertOptions,
): Promise<HalfBlockArt> {
  if (await detectChafa()) {
    try {
      return await renderWithChafa(filePath, {
        width: options?.maxWidth ?? 70,
        height: options?.maxHeight ?? 35,
      });
    } catch {
      // chafa failed, fall back
    }
  }
  return loadArtAsync(filePath, options);
}
