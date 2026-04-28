import type { ReasoningStep } from '@/lib/types'

export function ReasoningBlock({ steps }: { steps: ReasoningStep[] }) {
  return (
    <div
      className="my-2"
      style={{
        borderLeft: '2px solid var(--rule-mid)',
        paddingLeft: '12px',
        display: 'flex',
        flexDirection: 'column',
        gap: '5px',
      }}
    >
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', color: 'var(--ink-ghost)', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: '4px' }}>
        Reasoning
      </div>
      {steps.map((step) => (
        <div key={step.index} style={{ display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '10px',
            color: 'var(--ink-ghost)',
            flexShrink: 0,
            marginTop: '1px',
            minWidth: '16px',
          }}>
            {step.index}.
          </span>
          <span style={{ fontFamily: 'var(--font-display)', fontSize: '12px', color: 'var(--ink-muted)', lineHeight: 1.55 }}>
            {step.text}
          </span>
        </div>
      ))}
    </div>
  )
}
