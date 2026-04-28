interface Props {
  agents: { id: string; label: string }[]
  active: string
  onFilter: (id: string) => void
}

export function FilterBar({ agents, active, onFilter }: Props) {
  const items = [{ id: 'all', label: 'All' }, ...agents]

  return (
    <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '10px' }}>
      {items.map(({ id, label }) => {
        const isActive = active === id
        return (
          <button
            key={id}
            onClick={() => onFilter(id)}
            style={{
              padding: '4px 11px',
              fontFamily: 'var(--font-mono)',
              fontSize: '10px',
              letterSpacing: '0.03em',
              borderRadius: '7px',
              border: `1px solid ${isActive ? 'var(--ink-heavy)' : 'var(--rule-mid)'}`,
              background: isActive ? 'var(--ink-heavy)' : 'transparent',
              color: isActive ? '#fff' : 'var(--ink-muted)',
              cursor: 'pointer',
              transition: 'all 0.15s',
            }}
            onMouseEnter={(e) => {
              if (!isActive) {
                (e.currentTarget as HTMLElement).style.background = 'var(--s2)'
                ;(e.currentTarget as HTMLElement).style.color = 'var(--ink-heavy)'
              }
            }}
            onMouseLeave={(e) => {
              if (!isActive) {
                (e.currentTarget as HTMLElement).style.background = 'transparent'
                ;(e.currentTarget as HTMLElement).style.color = 'var(--ink-muted)'
              }
            }}
          >
            {label}
          </button>
        )
      })}
    </div>
  )
}
