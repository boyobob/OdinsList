import { useState, useEffect, useRef } from "react"
import { PixelArt } from "./PixelArt"
import { loadArtSmart } from "../utils/pixelArt"
import type { HalfBlockArt } from "../utils/pixelArt"
import { colors } from "../theme"

interface ImagePreviewProps {
  imagePath: string | null
  availableWidth: number
  availableHeight: number
}

export function ImagePreview({ imagePath, availableWidth, availableHeight }: ImagePreviewProps) {
  const [art, setArt] = useState<HalfBlockArt | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lastRenderKey = useRef("")
  const requestIdRef = useRef(0)
  const frameWidth = Math.max(10, availableWidth)
  const frameHeight = Math.max(5, availableHeight)

  useEffect(() => {
    if (!imagePath) {
      requestIdRef.current += 1
      lastRenderKey.current = ""
      setArt(null)
      setError(null)
      setLoading(false)
      return
    }

    const renderKey = `${imagePath}:${availableWidth}:${availableHeight}`
    if (renderKey === lastRenderKey.current) return

    if (debounceRef.current) {
      clearTimeout(debounceRef.current)
    }

    debounceRef.current = setTimeout(async () => {
      lastRenderKey.current = renderKey
      const requestId = ++requestIdRef.current
      setLoading(true)
      setError(null)

      try {
        const result = await loadArtSmart(imagePath, {
          maxWidth: frameWidth,
          maxHeight: frameHeight,
        })
        if (requestId !== requestIdRef.current) return
        setArt(result)
      } catch (e: any) {
        if (requestId !== requestIdRef.current) return
        setError(e.message ?? "Failed to render image")
        setArt(null)
      } finally {
        if (requestId !== requestIdRef.current) return
        setLoading(false)
      }
    }, 200)

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [imagePath, frameWidth, frameHeight])

  const frameProps = {
    width: frameWidth,
    height: frameHeight,
    justifyContent: "center" as const,
    alignItems: "center" as const,
    overflow: "hidden" as const,
  }

  if (!imagePath) {
    return (
      <box {...frameProps}>
        <text fg={colors.dimmed}>Waiting for image...</text>
      </box>
    )
  }

  if (loading && !art) {
    return (
      <box {...frameProps}>
        <text fg={colors.dimmed}>Loading preview...</text>
      </box>
    )
  }

  if (error) {
    return (
      <box {...frameProps}>
        <text fg={colors.red}>Preview error: {error}</text>
      </box>
    )
  }

  if (art) {
    return (
      <box {...frameProps}>
        <PixelArt art={art} maxWidth={frameWidth} maxHeight={frameHeight} />
      </box>
    )
  }

  return null
}
