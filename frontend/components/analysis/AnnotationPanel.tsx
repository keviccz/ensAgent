import type { AnnotationResult } from '@/lib/types'

export function AnnotationPanel({ annotation, loading }: { annotation: AnnotationResult | null; loading?: boolean }) {
  if (loading) {
    return (
      <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {[80, 60, 40, 100].map((w, i) => (
          <div key={i} className="skeleton" style={{ height: '14px', width: `${w}%` }} />
        ))}
      </div>
    )
  }

  if (!annotation) {
    return (
      <div style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '8px',
        padding: '24px',
        opacity: 0.5,
      }}>
        <svg viewBox="0 0 32 32" fill="none" width="24" height="24">
          <circle cx="16" cy="16" r="14" stroke="var(--ink-ghost)" strokeWidth="1.5" />
          <path d="M10 16 L16 10 L22 16" stroke="var(--ink-ghost)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          <line x1="16" y1="10" x2="16" y2="22" stroke="var(--ink-ghost)" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
        <span style={{ fontFamily: 'var(--font-display)', fontSize: '12px', color: 'var(--ink-muted)', textAlign: 'center' }}>
          Click a cluster to view annotation
        </span>
      </div>
    )
  }

  // confidence=0 means no annotation was found — show the interpretation message
  const hasData = annotation.confidence > 0 || !annotation.label.match(/^(Cluster|Domain) \d+$/)

  if (!hasData) {
    return (
      <div style={{
        padding: '20px', display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', gap: '8px', textAlign: 'center', opacity: 0.7,
      }}>
        <svg viewBox="0 0 24 24" fill="none" width="20" height="20">
          <circle cx="12" cy="12" r="10" stroke="var(--warn)" strokeWidth="1.5" />
          <path d="M12 8v4M12 16h.01" stroke="var(--warn)" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
        <span style={{ fontFamily: 'var(--font-display)', fontSize: '12px', color: 'var(--ink-muted)', lineHeight: 1.5 }}>
          {annotation.interpretation || 'Annotation not yet available. Run Stage D first.'}
        </span>
      </div>
    )
  }

  return (
    <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '14px', animation: 'fade-up 0.2s ease-out' }}>
      {/* Label */}
      <div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', letterSpacing: '0.07em', textTransform: 'uppercase', color: 'var(--ink-ghost)', marginBottom: '4px' }}>
          Annotation
        </div>
        <div style={{ fontFamily: 'var(--font-display)', fontSize: '15px', fontWeight: 600, letterSpacing: '-0.01em', color: 'var(--ink-heavy)' }}>
          {annotation.label}
        </div>
      </div>

      {/* Confidence */}
      <div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', letterSpacing: '0.07em', textTransform: 'uppercase', color: 'var(--ink-ghost)', marginBottom: '6px' }}>
          Confidence
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div className="progress-track" style={{ flex: 1 }}>
            <div
              className="progress-fill"
              style={{ width: `${annotation.confidence * 100}%`, background: 'var(--data)' }}
            />
          </div>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--data)', minWidth: '30px', textAlign: 'right' }}>
            {(annotation.confidence * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Marker genes */}
      <div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', letterSpacing: '0.07em', textTransform: 'uppercase', color: 'var(--ink-ghost)', marginBottom: '6px' }}>
          Marker Genes
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
          {annotation.markerGenes.map((g) => (
            <span
              key={g}
              style={{
                background: 'var(--data-dim)',
                color: 'var(--data)',
                fontFamily: 'var(--font-mono)',
                fontSize: '10.5px',
                padding: '2px 7px',
                borderRadius: '5px',
              }}
            >
              {g}
            </span>
          ))}
        </div>
      </div>

      {/* Interpretation */}
      <div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', letterSpacing: '0.07em', textTransform: 'uppercase', color: 'var(--ink-ghost)', marginBottom: '4px' }}>
          Interpretation
        </div>
        <p style={{ fontFamily: 'var(--font-display)', fontSize: '12px', color: 'var(--ink-muted)', lineHeight: 1.65 }}>
          {annotation.interpretation}
        </p>
      </div>
    </div>
  )
}
