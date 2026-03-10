import { useState } from "react"
import { useKeyboard } from "@opentui/react"
import type { Config } from "../types"
import { colors } from "../theme"

interface SetupViewProps {
  onComplete: (config: Partial<Config>) => void
}

interface SetupStep {
  key: keyof Config
  question: string
  hint: string
  defaultValue: string
}

const steps: SetupStep[] = [
  {
    key: "input_root_dir",
    question: "Where are your comic images stored?",
    hint: "(Folders should be named Box_01, Box_02, etc.)",
    defaultValue: "",
  },
  {
    key: "vlm_base_url",
    question: "What is your VLM API URL?",
    hint: "(Example: http://127.0.0.1:8000/v1 for local vLLM)",
    defaultValue: "",
  },
  {
    key: "vlm_model",
    question: "What VLM model should be used?",
    hint: "(Example: Qwen2.5-VL-32B)",
    defaultValue: "",
  },
  {
    key: "comicvine_api_key",
    question: "ComicVine API key?",
    hint: "(Optional, [Enter] Skip)",
    defaultValue: "",
  },
  {
    key: "gcd_db_path",
    question: "Path to GCD database?",
    hint: "(Optional, [Enter] Skip)",
    defaultValue: "",
  },
]

export function SetupView({ onComplete }: SetupViewProps) {
  const [stepIndex, setStepIndex] = useState(0)
  const [value, setValue] = useState(steps[0].defaultValue)
  const [answers, setAnswers] = useState<Partial<Config>>({})

  const step = steps[stepIndex]

  useKeyboard((key) => {
    if (key.name === "enter") {
      const newAnswers = { ...answers, [step.key]: value || step.defaultValue }
      setAnswers(newAnswers)

      if (stepIndex < steps.length - 1) {
        const nextStep = steps[stepIndex + 1]
        setValue(nextStep.defaultValue)
        setStepIndex(stepIndex + 1)
      } else {
        onComplete(newAnswers)
      }
    }
  })

  return (
    <box flexDirection="column" alignItems="center" justifyContent="center" flexGrow={1} gap={1}>
      <text fg={colors.accent}>
        <strong>Welcome to OdinsList</strong>
      </text>

      <box marginTop={2} flexDirection="column" gap={1}>
        <text fg={colors.white}>{step.question}</text>
        <text fg={colors.dimmed}>{step.hint}</text>
        <box marginTop={1}>
          <input
            value={value}
            onChange={setValue}
            placeholder="..."
            focused
            width={50}
          />
        </box>
      </box>

      <text fg={colors.dimmed} marginTop={2}>
        Step {stepIndex + 1} of {steps.length}
      </text>
    </box>
  )
}
