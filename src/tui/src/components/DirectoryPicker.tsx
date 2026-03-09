import { useState, useEffect, useCallback } from "react"
import { useRenderer, useTerminalDimensions } from "@opentui/react"
import { StatusBar } from "./StatusBar"
import { colors } from "../theme"
import type { Command, ScanEvent } from "../types"

interface DirectoryPickerProps {
  startPath: string
  sendCommand: (cmd: Command) => void
  onEvent: (listener: (event: ScanEvent) => void) => () => void
  onSelect: (path: string) => void
  onCancel: () => void
}

interface DirNode {
  name: string
  path: string
  children: DirNode[] | null  // null = not loaded, [] = empty
  expanded: boolean
  depth: number
}

export function DirectoryPicker({ startPath, sendCommand, onEvent, onSelect, onCancel }: DirectoryPickerProps) {
  const renderer = useRenderer()
  const { height } = useTerminalDimensions()
  const [nodes, setNodes] = useState<DirNode[]>([])
  const [cursorIndex, setCursorIndex] = useState(0)
  const [loading, setLoading] = useState(true)

  // Build flat visible list from tree
  const getVisibleNodes = useCallback((): DirNode[] => {
    const result: DirNode[] = []
    function walk(list: DirNode[]) {
      for (const node of list) {
        result.push(node)
        if (node.expanded && node.children) {
          walk(node.children)
        }
      }
    }
    walk(nodes)
    return result
  }, [nodes])

  const visible = getVisibleNodes()

  useEffect(() => {
    setNodes([])
    setCursorIndex(0)
    setLoading(true)

    const unsubscribe = onEvent((event) => {
      if (event.event !== "dirs") return
      const dirEvent = event as { event: "dirs"; path: string; dirs: string[] }

      if (dirEvent.path === startPath) {
        // Initial load
        const children = dirEvent.dirs.map((name): DirNode => ({
          name,
          path: `${startPath}/${name}`,
          children: null,
          expanded: false,
          depth: 0,
        }))
        setNodes(children)
        setLoading(false)
        return
      }

      // Expansion load — find the node and set its children
      setNodes(prev => {
        const updated = JSON.parse(JSON.stringify(prev)) as DirNode[]
        function findAndUpdate(list: DirNode[]): boolean {
          for (const node of list) {
            if (node.path === dirEvent.path) {
              node.children = dirEvent.dirs.map((name): DirNode => ({
                name,
                path: `${node.path}/${name}`,
                children: null,
                expanded: false,
                depth: node.depth + 1,
              }))
              node.expanded = true
              return true
            }
            if (node.children && findAndUpdate(node.children)) return true
          }
          return false
        }
        findAndUpdate(updated)
        return updated
      })
    })
    // Subscribe before requesting the first listing so fast backends cannot win the race.
    sendCommand({ cmd: "list-dirs", path: startPath })

    return unsubscribe
  }, [onEvent, sendCommand, startPath])

  // Keyboard handler
  useEffect(() => {
    const handler = (key: any) => {
      if (key.eventType === "release") return
      const vis = getVisibleNodes()

      if (key.name === "up") {
        setCursorIndex(i => Math.max(0, i - 1))
      } else if (key.name === "down") {
        setCursorIndex(i => Math.min(vis.length - 1, i + 1))
      } else if (key.name === "right") {
        // Expand
        const node = vis[cursorIndex]
        if (!node) return
        if (node.children === null) {
          sendCommand({ cmd: "list-dirs", path: node.path })
        } else if (!node.expanded) {
          setNodes(prev => {
            const updated = JSON.parse(JSON.stringify(prev)) as DirNode[]
            function find(list: DirNode[]): boolean {
              for (const n of list) {
                if (n.path === node.path) { n.expanded = true; return true }
                if (n.children && find(n.children)) return true
              }
              return false
            }
            find(updated)
            return updated
          })
        }
      } else if (key.name === "left") {
        // Collapse
        const node = vis[cursorIndex]
        if (!node) return
        if (node.expanded) {
          setNodes(prev => {
            const updated = JSON.parse(JSON.stringify(prev)) as DirNode[]
            function find(list: DirNode[]): boolean {
              for (const n of list) {
                if (n.path === node.path) { n.expanded = false; return true }
                if (n.children && find(n.children)) return true
              }
              return false
            }
            find(updated)
            return updated
          })
        }
      } else if (key.name === "return" || key.name === "enter") {
        const node = vis[cursorIndex]
        if (node) onSelect(node.path)
      } else if (key.name === "escape") {
        onCancel()
      }
    }

    renderer.keyInput.on("keypress", handler)
    return () => { renderer.keyInput.off("keypress", handler) }
  }, [cursorIndex, getVisibleNodes])

  if (loading) {
    return (
      <box flexDirection="column" width="100%" height="100%">
        <text fg={colors.dimmed}>Loading directories...</text>
      </box>
    )
  }

  // Render tree with adaptive viewport
  const maxVisible = Math.max(6, height - 8)
  const halfWindow = Math.floor(maxVisible / 2)
  let startIdx = Math.max(0, cursorIndex - halfWindow)
  const endIdx = Math.min(visible.length, startIdx + maxVisible)
  if (endIdx - startIdx < maxVisible) startIdx = Math.max(0, endIdx - maxVisible)

  return (
    <box flexDirection="column" width="100%" height="100%">
      <text fg={colors.white}>Select directory:</text>
      <text fg={colors.dimmed}>{startPath}</text>
      <box flexDirection="column" marginTop={1} flexGrow={1} border borderStyle="single" paddingX={1}>
        {visible.slice(startIdx, endIdx).map((node, i) => {
          const actualIdx = startIdx + i
          const selected = actualIdx === cursorIndex
          const indent = "  ".repeat(node.depth)
          const icon = node.expanded ? "v " : "> "
          return (
            <text key={node.path} fg={selected ? colors.accent : colors.fg} bg={selected ? colors.muted : undefined}>
              {indent}{icon}{node.name}/
            </text>
          )
        })}
      </box>
      <StatusBar
        left="[↑/↓] Navigate  ⋄  [→] Expand  ⋄  [←] Collapse"
        right="[Enter] Select  ⋄  [Esc] Cancel"
      />
    </box>
  )
}
