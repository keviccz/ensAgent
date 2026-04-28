'use client'
import { usePathname, useRouter } from 'next/navigation'
import { MessageSquare, BarChart2, Bot, Settings, Plus, Trash2 } from 'lucide-react'
import { useStore } from '@/lib/store'
import { useState } from 'react'

const NAV = [
  { id: 'chat',     label: 'Chat',     icon: MessageSquare, href: '/chat' },
  { id: 'analysis', label: 'Analysis', icon: BarChart2,     href: '/analysis' },
  { id: 'agents',   label: 'Agents',   icon: Bot,           href: '/agents' },
  { id: 'settings', label: 'Settings', icon: Settings,      href: '/settings' },
]

export function Sidebar() {
  const pathname = usePathname()
  const router = useRouter()
  const { conversations, activeConversationId, setActiveConversation, newConversation, deleteConversation } = useStore()
  const [hoveredConv, setHoveredConv] = useState<string | null>(null)

  const handleNew = () => {
    newConversation()
    router.push('/chat')
  }

  return (
    <aside
      aria-label="Primary navigation"
      className="flex flex-col h-screen bg-[#F7F7F7] border-r border-[rgba(0,0,0,0.07)]"
      style={{ width: 'var(--sidebar-w)', flexShrink: 0 }}
    >
      {/* Brand header */}
      <div
        className="h-12 flex items-center border-b border-[rgba(0,0,0,0.07)] gap-2.5 flex-shrink-0"
        style={{ padding: '0 20px' }}
      >
        {/* DNA-helix-inspired brand mark */}
        <div className="flex items-center gap-2.5" style={{ minWidth: 0 }}>
          <div className="w-5 h-5 relative flex-shrink-0">
            <svg viewBox="0 0 20 20" fill="none" className="w-full h-full" aria-hidden="true">
              <circle cx="10" cy="10" r="9" stroke="#0EA5E9" strokeWidth="1.5" />
              <path d="M6 7 Q10 10 14 7" stroke="#0EA5E9" strokeWidth="1.5" strokeLinecap="round" fill="none" />
              <path d="M6 13 Q10 10 14 13" stroke="#0EA5E9" strokeWidth="1.5" strokeLinecap="round" fill="none" />
              <circle cx="6" cy="7" r="1.5" fill="#0EA5E9" />
              <circle cx="14" cy="7" r="1.5" fill="#0EA5E9" />
              <circle cx="6" cy="13" r="1.5" fill="#0EA5E9" />
              <circle cx="14" cy="13" r="1.5" fill="#0EA5E9" />
            </svg>
          </div>
          <span style={{
            fontFamily: 'var(--font-display)',
            fontSize: '13px',
            fontWeight: 600,
            letterSpacing: '-0.02em',
            color: 'var(--ink-heavy)',
            whiteSpace: 'nowrap',
          }}>
            EnsAgent
          </span>
        </div>
      </div>

      {/* Nav */}
      <nav className="px-2 py-2 flex flex-col gap-0.5 flex-shrink-0" aria-label="Main sections">
        {NAV.map(({ id, label, icon: Icon, href }) => {
          const active = pathname.startsWith(`/${id}`)
          return (
            <button
              key={id}
              title={label}
              aria-label={label}
              onClick={() => router.push(href)}
              className="flex items-center gap-2.5 w-full text-left rounded-lg transition-all duration-150"
              style={{
                padding: '7px 10px',
                background: active ? '#E4E4E4' : 'transparent',
                color: active ? 'var(--ink-heavy)' : 'var(--ink-muted)',
                fontFamily: 'var(--font-display)',
                fontSize: '12.5px',
                fontWeight: active ? 500 : 400,
                border: 'none',
                cursor: 'pointer',
              }}
              onMouseEnter={(e) => {
                if (!active) (e.currentTarget as HTMLElement).style.background = 'rgba(0,0,0,0.04)'
              }}
              onMouseLeave={(e) => {
                if (!active) (e.currentTarget as HTMLElement).style.background = 'transparent'
              }}
            >
              <Icon
                size={13}
                strokeWidth={active ? 2 : 1.75}
                style={{ opacity: active ? 1 : 0.5, color: active ? 'var(--data)' : 'currentColor', flexShrink: 0 }}
              />
              {label}
            </button>
          )
        })}
      </nav>

      {/* Divider */}
      <div className="mx-3 h-px bg-[rgba(0,0,0,0.07)] flex-shrink-0" />

      {/* Conversation history */}
      <div className="flex-1 overflow-y-auto py-2 px-2 flex flex-col gap-0.5">
        {/* New conversation button */}
        <button
          aria-label="New conversation"
          title="New conversation"
          onClick={handleNew}
          className="flex items-center gap-2 w-full rounded-lg transition-colors duration-150"
          style={{
            padding: '6px 10px',
            fontFamily: 'var(--font-display)',
            fontSize: '11.5px',
            color: 'var(--ink-muted)',
            background: 'transparent',
            border: 'none',
            cursor: 'pointer',
            textAlign: 'left',
          }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(0,0,0,0.04)' }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'transparent' }}
        >
          <Plus size={11} strokeWidth={2} style={{ flexShrink: 0, opacity: 0.5 }} />
          New conversation
        </button>

        {conversations.length > 0 && (
          <div className="mt-1 flex flex-col gap-0.5">
            {conversations.slice(0, 12).map((conv) => {
              const isActive = conv.id === activeConversationId
              return (
                <div
                  key={conv.id}
                  className="relative flex items-center rounded-lg group"
                  style={{
                    background: isActive ? 'var(--s0)' : 'transparent',
                    border: isActive ? '1px solid rgba(0,0,0,0.07)' : '1px solid transparent',
                  }}
                  onMouseEnter={() => setHoveredConv(conv.id)}
                  onMouseLeave={() => setHoveredConv(null)}
                >
                  <button
                    onClick={() => { setActiveConversation(conv.id); router.push('/chat') }}
                    title={conv.title}
                    className="flex-1 text-left truncate transition-colors duration-150"
                    style={{
                      padding: '6px 10px',
                      fontFamily: 'var(--font-display)',
                      fontSize: '11.5px',
                      color: isActive ? 'var(--ink-heavy)' : 'var(--ink-muted)',
                      background: 'transparent',
                      border: 'none',
                      cursor: 'pointer',
                      maxWidth: '100%',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                    onMouseEnter={(e) => {
                      if (!isActive) (e.currentTarget.parentElement as HTMLElement).style.background = 'rgba(0,0,0,0.03)'
                    }}
                    onMouseLeave={(e) => {
                      if (!isActive) (e.currentTarget.parentElement as HTMLElement).style.background = 'transparent'
                    }}
                  >
                    {conv.title}
                  </button>

                  {hoveredConv === conv.id && (
                    <button
                      aria-label={`Delete conversation ${conv.title}`}
                      title="Delete conversation"
                      onClick={(e) => { e.stopPropagation(); deleteConversation(conv.id) }}
                      className="flex-shrink-0 flex items-center justify-center rounded transition-colors"
                      style={{
                        width: '22px',
                        height: '22px',
                        marginRight: '4px',
                        background: 'transparent',
                        border: 'none',
                        cursor: 'pointer',
                        color: 'var(--ink-ghost)',
                      }}
                      onMouseEnter={(e) => {
                        (e.currentTarget as HTMLElement).style.color = 'var(--err)'
                        ;(e.currentTarget as HTMLElement).style.background = 'rgba(239,68,68,0.08)'
                      }}
                      onMouseLeave={(e) => {
                        (e.currentTarget as HTMLElement).style.color = 'var(--ink-ghost)'
                        ;(e.currentTarget as HTMLElement).style.background = 'transparent'
                      }}
                    >
                      <Trash2 size={10} strokeWidth={1.75} aria-hidden="true" />
                    </button>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Footer: version tag */}
      <div
        className="border-t border-[rgba(0,0,0,0.07)] flex-shrink-0"
        style={{
          padding: '12px 16px',
          textAlign: 'left',
          fontFamily: 'var(--font-mono)',
          fontSize: '9.5px',
          color: 'var(--ink-ghost)',
          letterSpacing: '0.04em',
        }}
      >
        v0.1.0 · EnsAgent
      </div>
    </aside>
  )
}
