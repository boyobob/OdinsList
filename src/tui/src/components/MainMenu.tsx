import { useState } from "react"
import { useKeyboard } from "@opentui/react"
import { colors } from "../theme"
import { EyeSpinner, EYE_SPINNER_HEIGHT, EYE_SPINNER_WIDTH } from "./EyeSpinner"

export interface MenuItem {
  label: string
  value: string
}

interface MainMenuProps {
  items: MenuItem[]
  onSelect: (item: MenuItem) => void
}

export function MainMenu({ items, onSelect }: MainMenuProps) {
  const [selectedIndex, setSelectedIndex] = useState(0)

  useKeyboard((key) => {
    if (key.name === "up") {
      setSelectedIndex(i => (i - 1 + items.length) % items.length)
    }
    if (key.name === "down") {
      setSelectedIndex(i => (i + 1) % items.length)
    }
    if (key.name === "return") {
      onSelect(items[selectedIndex])
    }
  })

  return (
    <box flexDirection="column" alignItems="center">
      {items.map((item, i) => {
        const isSelected = i === selectedIndex
        return (
          <box
            key={item.value}
            flexDirection="row"
            alignItems="center"
            justifyContent="center"
            height={EYE_SPINNER_HEIGHT}
          >
            {isSelected ? (
              <box width={EYE_SPINNER_WIDTH + 1} height={EYE_SPINNER_HEIGHT} flexDirection="row">
                <EyeSpinner />
                <box width={1} />
              </box>
            ) : (
              <box width={EYE_SPINNER_WIDTH + 1} height={EYE_SPINNER_HEIGHT} />
            )}
            <text fg={isSelected ? colors.highlight : colors.dimmed}>{item.label}</text>
            {/* Mirror spacer on right to keep text centered */}
            <box width={EYE_SPINNER_WIDTH + 1} height={EYE_SPINNER_HEIGHT} />
          </box>
        )
      })}
    </box>
  )
}
