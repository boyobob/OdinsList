import { PNG } from "pngjs";
import { detectChafa, renderWithChafa } from "../utils/chafa";

const RESET = "\x1b[0m";

function hexToRgb(hex: string): [number, number, number] {
  const value = hex.startsWith("#") ? hex.slice(1) : hex;
  const r = Number.parseInt(value.slice(0, 2), 16);
  const g = Number.parseInt(value.slice(2, 4), 16);
  const b = Number.parseInt(value.slice(4, 6), 16);
  return [r, g, b];
}

function cellToAnsi(char: string, fg: string | null, bg: string | null): string {
  let out = "";
  if (fg) {
    const [r, g, b] = hexToRgb(fg);
    out += `\x1b[38;2;${r};${g};${b}m`;
  }
  if (bg) {
    const [r, g, b] = hexToRgb(bg);
    out += `\x1b[48;2;${r};${g};${b}m`;
  }
  return `${out}${char}${RESET}`;
}

function buildPreview(rows: Array<Array<{ char: string; fg: string | null; bg: string | null }>>): string {
  return rows
    .map((row) => row.map((cell) => cellToAnsi(cell.char, cell.fg, cell.bg)).join(""))
    .join("\n");
}

function hasVisibleContent(rows: Array<Array<{ char: string }>>): boolean {
  for (const row of rows) {
    for (const cell of row) {
      if (cell.char.trim() !== "") {
        return true;
      }
    }
  }
  return false;
}

async function createFixturePng(filePath: string): Promise<void> {
  const width = 64;
  const height = 32;
  const png = new PNG({ width, height });

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (width * y + x) * 4;

      const r = Math.round((x / (width - 1)) * 255);
      const g = Math.round((y / (height - 1)) * 255);
      const b = Math.round(((x + y) / (width + height - 2)) * 255);

      const cx = width / 2;
      const cy = height / 2;
      const dist = Math.hypot(x - cx, y - cy);
      const ring = dist > 8 && dist < 12;

      png.data[idx] = ring ? 255 : r;
      png.data[idx + 1] = ring ? 255 : g;
      png.data[idx + 2] = ring ? 255 : b;
      png.data[idx + 3] = 255;
    }
  }

  await Bun.write(filePath, PNG.sync.write(png));
}

async function main(): Promise<void> {
  const providedImage = Bun.argv[2];
  const fixturePath = "/tmp/chafa-fixture.png";
  const imagePath = providedImage ?? fixturePath;

  if (!(await detectChafa())) {
    console.error("chafa binary was not found in PATH. Install chafa and retry.");
    process.exit(1);
  }

  if (!providedImage) {
    await createFixturePng(imagePath);
  }

  const art = await renderWithChafa(imagePath, { width: 40, height: 20 });

  if (art.width <= 0 || art.height <= 0) {
    throw new Error(`Expected non-empty art dimensions, got ${art.width}x${art.height}`);
  }

  if (!hasVisibleContent(art.rows)) {
    throw new Error("Rendered output had no visible characters.");
  }

  const preview = buildPreview(art.rows);
  console.log(`Source image: ${imagePath}`);
  console.log(`Rendered size: ${art.width}x${art.height}`);
  console.log("\nANSI Preview:\n");
  console.log(preview);
  console.log("\nPASS: chafa.ts successfully converted image data to ANSI rows.");
}

await main();
