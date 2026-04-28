interface TopbarProps {
  title: string
  subtitle?: string
  showDot?: boolean
  actions?: React.ReactNode
}

export function Topbar({ title, subtitle, showDot, actions }: TopbarProps) {
  return (
    <header
      className="flex items-center gap-2.5 border-b flex-shrink-0"
      style={{
        height: '46px',
        padding: '0 24px',
        borderColor: 'var(--rule)',
        background: 'var(--s0)',
      }}
    >
      <span style={{
        fontFamily: 'var(--font-display)',
        fontSize: '13px',
        fontWeight: 600,
        letterSpacing: '-0.01em',
        color: 'var(--ink-heavy)',
      }}>
        {title}
      </span>

      {showDot && (
        <span
          className="animate-pulse-dot"
          style={{
            display: 'inline-block',
            width: '5px',
            height: '5px',
            borderRadius: '50%',
            background: 'var(--data)',
            flexShrink: 0,
          }}
        />
      )}

      {subtitle && (
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '10.5px',
          color: 'var(--ink-ghost)',
          letterSpacing: '0.01em',
        }}>
          {subtitle}
        </span>
      )}

      {actions && <div className="ml-auto flex items-center gap-2">{actions}</div>}
    </header>
  )
}
