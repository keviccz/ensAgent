'use client'
import { useState, useRef, KeyboardEvent, useEffect } from 'react'
import { ArrowUp, Square } from 'lucide-react'

interface Props {
  onSend: (text: string) => void
  onStop?: () => void
  isStreaming?: boolean
  disabled?: boolean
}

export function ChatInput({ onSend, onStop, isStreaming, disabled }: Props) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const adjustHeight = () => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 128)}px`
  }

  useEffect(() => { adjustHeight() }, [value])

  const handleSend = () => {
    const text = value.trim()
    if (!text || disabled) return
    // If streaming, stop first then send
    if (isStreaming && onStop) onStop()
    onSend(text)
    setValue('')
  }

  const handleKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const hasText = value.trim().length > 0
  const canSend = hasText && !disabled

  return (
    <div style={{ padding: '0 20px 14px', flexShrink: 0 }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-end',
          gap: '8px',
          border: `1px solid ${(canSend || isStreaming) ? 'var(--rule-mid)' : 'var(--rule)'}`,
          borderRadius: '14px',
          padding: '10px 10px 10px 14px',
          background: 'var(--s1)',
          transition: 'border-color 0.15s, box-shadow 0.15s',
          boxShadow: (canSend || isStreaming) ? '0 2px 8px rgba(0,0,0,0.05)' : 'none',
        }}
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKey}
          placeholder={isStreaming ? 'Type to interrupt agent…' : 'Message EnsAgent…'}
          rows={1}
          disabled={disabled && !isStreaming}
          style={{
            flex: 1,
            background: 'transparent',
            border: 'none',
            outline: 'none',
            resize: 'none',
            fontFamily: 'var(--font-display)',
            fontSize: '12.5px',
            color: 'var(--ink-heavy)',
            lineHeight: 1.55,
            minHeight: '22px',
            maxHeight: '128px',
            overflowY: 'auto',
          }}
        />

        {/* Stop button — shown when streaming */}
        {isStreaming && !hasText && (
          <button
            onClick={() => onStop?.()}
            title="Stop agent"
            style={{
              width: '30px',
              height: '30px',
              borderRadius: '8px',
              background: 'var(--s2)',
              border: '1px solid var(--rule-mid)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              flexShrink: 0,
              transition: 'background 0.15s',
            }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(239,68,68,0.08)' }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'var(--s2)' }}
          >
            <Square size={11} strokeWidth={2} color="var(--err)" fill="var(--err)" />
          </button>
        )}

        {/* Send button */}
        {(!isStreaming || hasText) && (
          <button
            onClick={handleSend}
            disabled={!canSend}
            style={{
              width: '30px',
              height: '30px',
              borderRadius: '8px',
              background: canSend ? 'var(--ctrl)' : 'var(--s3)',
              border: 'none',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: canSend ? 'pointer' : 'default',
              flexShrink: 0,
              transition: 'background 0.15s, transform 0.1s',
            }}
            onMouseEnter={(e) => { if (canSend) (e.currentTarget as HTMLElement).style.background = 'var(--ctrl-h)' }}
            onMouseLeave={(e) => { if (canSend) (e.currentTarget as HTMLElement).style.background = 'var(--ctrl)' }}
            onMouseDown={(e) => { if (canSend) (e.currentTarget as HTMLElement).style.transform = 'scale(0.93)' }}
            onMouseUp={(e) => { (e.currentTarget as HTMLElement).style.transform = 'scale(1)' }}
          >
            <ArrowUp
              size={14}
              strokeWidth={2.5}
              color={canSend ? '#fff' : 'var(--ink-ghost)'}
            />
          </button>
        )}
      </div>

      <div style={{
        textAlign: 'center',
        marginTop: '8px',
        fontFamily: 'var(--font-mono)',
        fontSize: '9.5px',
        color: 'var(--ink-ghost)',
        letterSpacing: '0.02em',
      }}>
        {isStreaming
          ? 'Agent running — type to interrupt · Stop ⬛ to cancel'
          : 'Enter to send · Shift+Enter for new line'}
      </div>
    </div>
  )
}
