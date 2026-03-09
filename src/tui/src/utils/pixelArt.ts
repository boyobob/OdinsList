import { PNG } from "pngjs";
import { detectChafa, renderWithChafa } from "./chafa";

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
  contrastStretch?: boolean; // remap luma range to fill 30-230 (default true)
}

// ── Constants ──────────────────────────────────────────────────────────────────

const UPPER_HALF = "\u2580"; // ▀
const LOWER_HALF = "\u2584"; // ▄
const FULL_BLOCK = "\u2588"; // █
const SPACE = " ";

// ── Helpers ────────────────────────────────────────────────────────────────────

/** ITU-R 601-2 luma: convert RGB to greyscale brightness. */
function toLuma(r: number, g: number, b: number): number {
  return Math.round(0.299 * r + 0.587 * g + 0.114 * b);
}

/** Format a greyscale value as an ANSI-compatible hex colour string. */
function greyHex(luma: number): string {
  const clamped = Math.max(0, Math.min(255, luma));
  const h = clamped.toString(16).padStart(2, "0");
  return `#${h}${h}${h}`;
}

// ── Resize (nearest-neighbour) ─────────────────────────────────────────────────

interface RawImage {
  data: Buffer;
  width: number;
  height: number;
}

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

// ── Contrast stretching ────────────────────────────────────────────────────────

/** Find min/max luma of opaque pixels and build a remap function. */
function buildContrastMap(
  img: RawImage,
  alphaThreshold: number,
): (luma: number) => number {
  let minLuma = 255;
  let maxLuma = 0;

  for (let i = 0; i < img.data.length; i += 4) {
    if (img.data[i + 3] < alphaThreshold) continue;
    const l = toLuma(img.data[i], img.data[i + 1], img.data[i + 2]);
    if (l < minLuma) minLuma = l;
    if (l > maxLuma) maxLuma = l;
  }

  const range = maxLuma - minLuma;
  if (range < 10) {
    // Not enough range to stretch meaningfully
    return (luma: number) => luma;
  }

  // Remap [minLuma, maxLuma] → [30, 230] to keep visibility on dark backgrounds
  const outMin = 30;
  const outMax = 230;
  return (luma: number) => {
    const normalized = (luma - minLuma) / range;
    return Math.round(outMin + normalized * (outMax - outMin));
  };
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
  const shouldStretchContrast = options?.contrastStretch !== false; // default true

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

  // Build contrast map after resize (operates on final pixel data)
  const remapLuma = shouldStretchContrast
    ? buildContrastMap(img, alphaThreshold)
    : (l: number) => l;

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

      const topLuma = remapLuma(toLuma(topR, topG, topB));
      const botLuma = remapLuma(toLuma(botR, botG, botB));

      if (topTransparent && botTransparent) {
        cells.push({ char: SPACE, fg: null, bg: null });
      } else if (topTransparent) {
        cells.push({ char: LOWER_HALF, fg: greyHex(botLuma), bg: null });
      } else if (botTransparent) {
        cells.push({ char: UPPER_HALF, fg: greyHex(topLuma), bg: null });
      } else {
        if (topLuma === botLuma) {
          cells.push({ char: FULL_BLOCK, fg: greyHex(topLuma), bg: null });
        } else {
          cells.push({
            char: UPPER_HALF,
            fg: greyHex(topLuma),
            bg: greyHex(botLuma),
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

export async function loadArtAsync(
  filePath: string,
  options?: ConvertOptions,
): Promise<HalfBlockArt> {
  const file = Bun.file(filePath);
  const arrayBuffer = await file.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);
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
