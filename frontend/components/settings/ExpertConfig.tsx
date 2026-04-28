'use client'
import type { AnnotationExpertConfig } from '@/lib/types'

function FieldLabel({ children }: { children: React.ReactNode }) {
  return (
    <label style={{
      fontFamily: 'var(--font-mono)', fontSize: '9.5px', letterSpacing: '0.07em',
      textTransform: 'uppercase', color: 'var(--ink-ghost)', display: 'block', marginBottom: '5px',
    }}>
      {children}
    </label>
  )
}

function WeightSlider({
  label, desc, value, onChange,
}: { label: string; desc: string; value: number; onChange: (v: number) => void }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <label style={{ fontFamily: 'var(--font-display)', fontSize: '12.5px', color: 'var(--ink-mid)' }}>
          {label}
        </label>
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: '10.5px', color: 'var(--ink-mid)',
          background: 'var(--s2)', padding: '1px 6px', borderRadius: '4px',
          border: '1px solid var(--rule)',
        }}>
          {value.toFixed(2)}
        </span>
      </div>
      <p style={{ fontFamily: 'var(--font-display)', fontSize: '11px', color: 'var(--ink-ghost)', margin: 0, lineHeight: 1.4 }}>
        {desc}
      </p>
      <input
        type="range"
        min={0} max={1} step={0.05}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ width: '100%', accentColor: 'var(--data)', cursor: 'pointer', marginTop: '2px' }}
      />
    </div>
  )
}

interface Props {
  config: Partial<AnnotationExpertConfig>
  onChange: (key: keyof AnnotationExpertConfig, value: number | boolean) => void
}

export function ExpertConfigPanel({ config, onChange }: Props) {
  const wMarker  = config.annotationWMarker  ?? 0.30
  const wPathway = config.annotationWPathway ?? 0.20
  const wSpatial = config.annotationWSpatial ?? 0.30
  const wVlm     = config.annotationWVlm     ?? 0.20

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
      {/* Expert weights */}
      <div>
        <FieldLabel>Expert Weights</FieldLabel>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '5px' }}>
          <WeightSlider
            label="Marker"
            desc="Marker gene expression alignment with known cell types"
            value={wMarker}
            onChange={(v) => onChange('annotationWMarker', v)}
          />
          <WeightSlider
            label="Pathway"
            desc="Gene set enrichment and pathway activity score"
            value={wPathway}
            onChange={(v) => onChange('annotationWPathway', v)}
          />
          <WeightSlider
            label="Spatiality"
            desc="Spatial coherence and neighborhood context score"
            value={wSpatial}
            onChange={(v) => onChange('annotationWSpatial', v)}
          />
          <WeightSlider
            label="Visual (VLM)"
            desc="Morphological evidence from vision-language model"
            value={wVlm}
            onChange={(v) => onChange('annotationWVlm', v)}
          />
        </div>
      </div>

      <div style={{ height: '1px', background: 'var(--rule)' }} />

      {/* Critic config */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <FieldLabel>Critic Agent</FieldLabel>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <label style={{ fontFamily: 'var(--font-display)', fontSize: '12.5px', color: 'var(--ink-mid)' }}>
              Standard Score Threshold
            </label>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10.5px', color: 'var(--ink-mid)', background: 'var(--s2)', padding: '1px 6px', borderRadius: '4px', border: '1px solid var(--rule)' }}>
              {(config.annotationStandardScore ?? 0.65).toFixed(2)}
            </span>
          </div>
          <p style={{ fontFamily: 'var(--font-display)', fontSize: '11px', color: 'var(--ink-ghost)', margin: 0, lineHeight: 1.4 }}>
            Minimum weighted score required to accept an annotation
          </p>
          <input
            type="range" min={0} max={1} step={0.05}
            value={config.annotationStandardScore ?? 0.65}
            onChange={(e) => onChange('annotationStandardScore', parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--data)', cursor: 'pointer', marginTop: '2px' }}
          />
        </div>

        <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={config.annotationVlmRequired ?? true}
            onChange={(e) => onChange('annotationVlmRequired', e.target.checked)}
            style={{ accentColor: 'var(--ink-heavy)', cursor: 'pointer', width: '14px', height: '14px' }}
          />
          <div>
            <span style={{ fontFamily: 'var(--font-display)', fontSize: '12.5px', color: 'var(--ink-mid)' }}>
              VLM Required
            </span>
            <p style={{ fontFamily: 'var(--font-display)', fontSize: '11px', color: 'var(--ink-ghost)', margin: '2px 0 0', lineHeight: 1.4 }}>
              Reject annotation if VLM score is absent
            </p>
          </div>
        </label>
      </div>

      <div style={{ height: '1px', background: 'var(--rule)' }} />

      {/* Proposer config */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <FieldLabel>Proposer Agent</FieldLabel>

        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '2px' }}>
            <label style={{ fontFamily: 'var(--font-display)', fontSize: '12.5px', color: 'var(--ink-mid)', flex: 1 }}>
              Max Rounds
            </label>
            <input
              type="number"
              min={1} max={10}
              value={config.annotationMaxRounds ?? 3}
              onChange={(e) => onChange('annotationMaxRounds', parseInt(e.target.value, 10))}
              className="input-base"
              style={{ width: '64px', textAlign: 'center' }}
            />
          </div>
          <p style={{ fontFamily: 'var(--font-display)', fontSize: '11px', color: 'var(--ink-ghost)', margin: '4px 0 0', lineHeight: 1.4 }}>
            Maximum Proposer–Critic debate rounds per domain
          </p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <label style={{ fontFamily: 'var(--font-display)', fontSize: '12.5px', color: 'var(--ink-mid)' }}>
              VLM Min Score
            </label>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10.5px', color: 'var(--ink-mid)', background: 'var(--s2)', padding: '1px 6px', borderRadius: '4px', border: '1px solid var(--rule)' }}>
              {(config.annotationVlmMinScore ?? 0.30).toFixed(2)}
            </span>
          </div>
          <p style={{ fontFamily: 'var(--font-display)', fontSize: '11px', color: 'var(--ink-ghost)', margin: 0, lineHeight: 1.4 }}>
            Minimum VLM agent score to count as valid evidence
          </p>
          <input
            type="range" min={0} max={1} step={0.05}
            value={config.annotationVlmMinScore ?? 0.30}
            onChange={(e) => onChange('annotationVlmMinScore', parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--data)', cursor: 'pointer', marginTop: '2px' }}
          />
        </div>
      </div>
    </div>
  )
}
