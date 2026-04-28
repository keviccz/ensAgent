import type { StageState } from '@/lib/types'

const STATUS_COLOR: Record<string, string> = {
  idle:    'var(--s3)',
  running: 'var(--data)',
  done:    'var(--ok)',
  error:   'var(--err)',
  skipped: 'var(--ink-ghost)',
}

const STATUS_LABEL: Record<string, string> = {
  idle:    'idle',
  running: 'running',
  done:    'done',
  error:   'error',
  skipped: 'skipped',
}

export function PipelineProgress({ stages }: { stages: StageState[] }) {
  const anyActive = stages.some((s) => s.status === 'running')

  return (
    <div
      style={{
        margin: '0 20px 12px',
        padding: '12px 14px',
        background: 'var(--s1)',
        border: '1px solid var(--rule)',
        borderRadius: '12px',
        flexShrink: 0,
      }}
    >
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        marginBottom: '10px',
      }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--ink-ghost)' }}>
          Pipeline
        </span>
        {anyActive && (
          <span style={{
            display: 'inline-block',
            width: '5px',
            height: '5px',
            borderRadius: '50%',
            background: 'var(--data)',
            animation: 'pulse-dot 1.5s ease-in-out infinite',
          }} />
        )}
      </div>

      {/* Stages */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
        {stages.map((stage) => (
          <div key={stage.name}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
              <span style={{ fontFamily: 'var(--font-display)', fontSize: '11px', color: 'var(--ink-mid)' }}>
                {stage.label}
              </span>
              <span style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '9px',
                color: stage.status === 'running' ? 'var(--data)' : stage.status === 'done' ? 'var(--ok)' : 'var(--ink-ghost)',
                letterSpacing: '0.03em',
              }}>
                {STATUS_LABEL[stage.status]}
              </span>
            </div>
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{
                  width: `${stage.status === 'done' ? 100 : stage.status === 'skipped' ? 100 : stage.progress}%`,
                  background: STATUS_COLOR[stage.status],
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
