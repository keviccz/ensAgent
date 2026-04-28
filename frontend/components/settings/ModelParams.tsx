'use client'

interface SliderProps {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
  hint?: string
}

function Slider({ label, value, min, max, step, onChange, hint }: SliderProps) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <span style={{ fontFamily: 'var(--font-display)', fontSize: '12.5px', color: 'var(--ink-mid)' }}>
          {label}
        </span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--ink-heavy)', fontWeight: 500 }}>
          {value.toFixed(2)}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ width: '100%', accentColor: 'var(--ink-heavy)', cursor: 'pointer' }}
      />
      {hint && (
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', color: 'var(--ink-ghost)' }}>
          {hint}
        </span>
      )}
    </div>
  )
}

interface Props {
  temperature: number
  topP: number
  visualFactor: number
  onChange: (key: string, value: number) => void
}

export function ModelParams({ temperature, topP, visualFactor, onChange }: Props) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <Slider
        label="Temperature"
        value={temperature}
        min={0} max={2} step={0.01}
        onChange={(v) => onChange('temperature', v)}
        hint="Controls output randomness. 0 = deterministic, 2 = very creative"
      />
      <Slider
        label="Top-p"
        value={topP}
        min={0} max={1} step={0.01}
        onChange={(v) => onChange('topP', v)}
        hint="Nucleus sampling threshold. Lower = more focused"
      />
      <Slider
        label="Visual Factor"
        value={visualFactor}
        min={0} max={1} step={0.01}
        onChange={(v) => onChange('visualFactor', v)}
        hint="Weight of visual (VLM) scoring vs text scoring in Stage B"
      />
    </div>
  )
}
