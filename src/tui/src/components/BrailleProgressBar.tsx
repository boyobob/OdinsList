import { useState, useEffect } from "react"
import { colors } from "../theme"

interface BrailleProgressBarProps {
  progress: number  // 0-100
  width?: number
  playing?: boolean
}

const EMPTY_CELL = "\u2800"
const FULL_CELL = "\u28FF"
const FILL_CHARS = [EMPTY_CELL, "\u2801", "\u2803", "\u2807", "\u280F", "\u281F", "\u283F", "\u287F", "\u28F7", "\u28EF", "\u28DF", FULL_CELL]
// A short wave that keeps the frontier alive even when determinate progress updates are sparse.
const WAKE_WAVE = ["\u281F", "\u283F", "\u287F", "\u28F7", "\u28EF", "\u28DF", FULL_CELL, "\u28DF", "\u28EF", "\u28F7", "\u287F", "\u283F"]
const HEAD_PULSE = [-1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5]
const ANIMATION_INTERVAL = 100
const DISPLAY_EASING = 0.22
const MIN_PROGRESS_STEP = 0.08

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function charForFill(fill: number): string {
  if (fill <= 0) return EMPTY_CELL
  if (fill >= 1) return FULL_CELL

  const index = clamp(Math.round(fill * (FILL_CHARS.length - 1)), 1, FILL_CHARS.length - 2)
  return FILL_CHARS[index] ?? FULL_CELL
}

function animatedHeadFill(fill: number, animationFrame: number, offset: number): number {
  if (fill <= 0 || fill >= 1) return fill

  const pulse = HEAD_PULSE[(animationFrame + offset) % HEAD_PULSE.length] ?? 0
  const amplitude = Math.min(fill, 1 - fill, 0.14)
  return clamp(fill + pulse * amplitude, 0, 1)
}

function buildBar(progress: number, width: number, animationFrame: number): string {
  const filledExact = clamp((progress / 100) * width, 0, width)
  if (filledExact <= 0) return EMPTY_CELL.repeat(width)
  if (filledExact >= width) return FULL_CELL.repeat(width)

  const activeCells = Math.ceil(filledExact)
  const wakeLength = clamp(Math.round(width * 0.12), 4, 6)
  const wakeStart = Math.max(0, activeCells - wakeLength)
  let bar = ""

  for (let index = 0; index < width; index += 1) {
    const cellFill = clamp(filledExact - index, 0, 1)
    if (cellFill <= 0) {
      bar += EMPTY_CELL
      continue
    }

    if (index < wakeStart) {
      bar += FULL_CELL
      continue
    }

    const localOffset = index - wakeStart
    if (cellFill < 1) {
      bar += charForFill(animatedHeadFill(cellFill, animationFrame, localOffset))
      continue
    }

    bar += WAKE_WAVE[(animationFrame + localOffset) % WAKE_WAVE.length] ?? FULL_CELL
  }

  return bar
}

export function BrailleProgressBar({ progress, width = 40, playing = true }: BrailleProgressBarProps) {
  const targetProgress = clamp(progress, 0, 100)
  const [displayProgress, setDisplayProgress] = useState(targetProgress)
  const [animationFrame, setAnimationFrame] = useState(0)

  useEffect(() => {
    setDisplayProgress((current) => {
      if (!playing || targetProgress < current) return targetProgress
      return current
    })
  }, [playing, targetProgress])

  useEffect(() => {
    if (!playing) return
    const interval = setInterval(() => {
      setAnimationFrame((frame) => (frame + 1) % WAKE_WAVE.length)
      setDisplayProgress((current) => {
        if (targetProgress <= current) return current

        const delta = targetProgress - current
        if (delta < 0.05) return targetProgress

        return Math.min(targetProgress, current + Math.max(delta * DISPLAY_EASING, MIN_PROGRESS_STEP))
      })
    }, ANIMATION_INTERVAL)
    return () => clearInterval(interval)
  }, [playing, targetProgress])

  const bar = buildBar(displayProgress, width, animationFrame)
  const pct = `${Math.round(targetProgress)}%`

  return (
    <box flexDirection="row" gap={1}>
      <text fg={colors.accent}>{bar}</text>
      <text fg={colors.dimmed}>{pct}</text>
    </box>
  )
}
