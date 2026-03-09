import { expect, test, describe } from "bun:test";
import { parseAnsiArt } from "./chafa";

describe("parseAnsiArt", () => {
  test("ignores non-SGR CSI sequences such as cursor visibility toggles", () => {
    const rows = parseAnsiArt("\x1b[?25l\x1b[38;2;255;0;0mA\x1b[0m\x1b[?25h");

    expect(rows).toHaveLength(1);
    expect(rows[0]).toEqual([{ char: "A", fg: "#ff0000", bg: null }]);
  });

  test("parses truecolor foreground and background SGR sequences", () => {
    const rows = parseAnsiArt("\x1b[38;2;1;2;3m\x1b[48;2;4;5;6mZ\x1b[0m");

    expect(rows).toHaveLength(1);
    expect(rows[0]).toEqual([{ char: "Z", fg: "#010203", bg: "#040506" }]);
  });
});
