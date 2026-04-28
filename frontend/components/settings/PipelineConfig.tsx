'use client'
import { useEffect, useState } from 'react'
import { getAvailableMethods } from '@/lib/api'
import type { PipelineConfig } from '@/lib/types'

const FALLBACK_METHODS = ['IRIS', 'BASS', 'DR-SC', 'BayesSpace', 'SEDR', 'GraphST', 'STAGATE', 'stLearn']

function FieldLabel({ children }: { children: React.ReactNode }) {
  return (
    <label style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', letterSpacing: '0.07em', textTransform: 'uppercase', color: 'var(--ink-ghost)', display: 'block', marginBottom: '5px' }}>
      {children}
    </label>
  )
}

interface Props {
  config: Partial<PipelineConfig>
  onChange: (key: keyof PipelineConfig, value: unknown) => void
}

export function PipelineConfigPanel({ config, onChange }: Props) {
  const [availableMethods, setAvailableMethods] = useState<string[]>(FALLBACK_METHODS)

  useEffect(() => {
    getAvailableMethods()
      .then((res) => { if (res.methods?.length > 0) setAvailableMethods(res.methods) })
      .catch(() => { /* keep fallback */ })
  }, [])

  const methods = config.methods ?? availableMethods

  const toggleMethod = (m: string) => {
    const next = methods.includes(m) ? methods.filter((x) => x !== m) : [...methods, m]
    onChange('methods', next)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
      {/* Data path */}
      <div>
        <FieldLabel>Data Path</FieldLabel>
        <input
          type="text"
          className="input-base"
          value={config.dataPath ?? ''}
          onChange={(e) => onChange('dataPath', e.target.value)}
          placeholder="Tool-runner/151507"
        />
      </div>

      {/* Sample ID */}
      <div>
        <FieldLabel>Sample ID</FieldLabel>
        <input
          type="text"
          className="input-base"
          value={config.sampleId ?? ''}
          onChange={(e) => onChange('sampleId', e.target.value)}
          placeholder="DLPFC_151507"
        />
      </div>

      {/* N clusters */}
      <div>
        <FieldLabel>N Clusters</FieldLabel>
        <input
          type="number"
          className="input-base"
          value={config.nClusters ?? 7}
          min={2} max={20}
          onChange={(e) => onChange('nClusters', parseInt(e.target.value, 10))}
          style={{ width: '80px' }}
        />
      </div>

      {/* Methods */}
      <div>
        <FieldLabel>Clustering Methods</FieldLabel>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
          {availableMethods.map((m) => {
            const checked = methods.includes(m)
            return (
              <label
                key={m}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '7px',
                  cursor: 'pointer',
                  padding: '5px 8px',
                  borderRadius: '7px',
                  background: checked ? 'var(--data-dim)' : 'var(--s2)',
                  border: `1px solid ${checked ? 'rgba(14,165,233,0.2)' : 'transparent'}`,
                  transition: 'all 0.15s',
                }}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => toggleMethod(m)}
                  style={{ accentColor: 'var(--data)', cursor: 'pointer' }}
                />
                <span style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '11px',
                  color: checked ? 'var(--data)' : 'var(--ink-mid)',
                }}>
                  {m}
                </span>
              </label>
            )
          })}
        </div>
      </div>

      {/* Skip stages */}
      <div>
        <FieldLabel>Skip Stages</FieldLabel>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {([
            { key: 'skipToolRunner' as const, label: 'Skip Tool-Runner (Stage A)' },
            { key: 'skipScoring' as const,    label: 'Skip Scoring (Stage B)' },
          ]).map(({ key, label }) => (
            <label key={key} style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={!!(config[key])}
                onChange={(e) => onChange(key, e.target.checked)}
                style={{ accentColor: 'var(--ink-heavy)', cursor: 'pointer' }}
              />
              <span style={{ fontFamily: 'var(--font-display)', fontSize: '12.5px', color: 'var(--ink-mid)' }}>
                {label}
              </span>
            </label>
          ))}
        </div>
      </div>
    </div>
  )
}
