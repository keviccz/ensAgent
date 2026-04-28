'use client'
import { useEffect, useRef } from 'react'
import type { ChatMessage, ToolCall, ChartData } from '@/lib/types'
import { MessageBubble } from './MessageBubble'
import { ToolCallBlock } from './ToolCallBlock'
import { ChatChart } from './ChatChart'

interface Props {
  messages: ChatMessage[]
  streamingText?: string
  streamingToolCalls?: ToolCall[]
  streamingImages?: string[]
  streamingCharts?: ChartData[]
  isPending?: boolean
}

export function ChatMessages({ messages, streamingText, streamingToolCalls = [], streamingImages = [], streamingCharts = [], isPending = false }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, streamingText])

  return (
    <div
      style={{
        flex: 1,
        overflowY: 'auto',
        padding: '16px 20px 12px',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {messages.length === 0 && !streamingText && (
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '20px',
          opacity: 0.9,
          animation: 'fade-in 0.4s ease-out',
        }}>
          {/* Minimal DNA motif */}
          <svg viewBox="0 0 48 48" fill="none" width="52" height="52">
            <path d="M12 10 Q24 24 36 10" stroke="var(--data)" strokeWidth="2" strokeLinecap="round" fill="none" />
            <path d="M12 38 Q24 24 36 38" stroke="var(--data)" strokeWidth="2" strokeLinecap="round" fill="none" />
            <line x1="24" y1="10" x2="24" y2="38" stroke="var(--rule-mid)" strokeWidth="1.5" strokeDasharray="3 3" />
          </svg>
          <span style={{
            fontFamily: 'var(--font-display)',
            fontSize: '17px',
            fontWeight: 600,
            color: 'var(--ink-heavy)',
            textAlign: 'center',
            lineHeight: 1.5,
          }}>
            Start a conversation with EnsAgent
          </span>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', textAlign: 'center' }}>
            {[
              'Run the full spatial analysis pipeline',
              'Check clustering results for DLPFC_151507',
              'Annotate domain 3 using LLM agents',
            ].map((hint) => (
              <span key={hint} style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '13px',
                color: 'var(--ink-ghost)',
              }}>
                "{hint}"
              </span>
            ))}
          </div>
        </div>
      )}

      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}

      {/* Streaming tool calls */}
      {streamingToolCalls.length > 0 && (
        <div style={{ marginBottom: '8px' }}>
          {streamingToolCalls.map((tc) => (
            <ToolCallBlock key={tc.id} call={tc} />
          ))}
        </div>
      )}

      {/* Streaming images from tool results */}
      {streamingImages.map((b64, i) => (
        <div key={i} style={{ marginBottom: '10px', maxWidth: '460px', animation: 'fade-in 0.3s ease-out' }}>
          <div style={{ borderRadius: '8px', overflow: 'hidden', border: '1px solid var(--rule-mid)', background: 'var(--s1)' }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`data:image/png;base64,${b64}`}
              alt="Clustering plot"
              style={{ display: 'block', width: '100%', height: 'auto', maxHeight: '340px', objectFit: 'contain', cursor: 'zoom-in' }}
              onClick={() => window.open(`data:image/png;base64,${b64}`, '_blank')}
            />
            <div style={{ padding: '4px 10px', borderTop: '1px solid var(--rule)', fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--ink-ghost)', background: 'var(--s2)' }}>
              Click to open full size
            </div>
          </div>
        </div>
      ))}

      {/* Streaming charts */}
      {streamingCharts.map((chart, i) => (
        <div key={i} style={{ maxWidth: '78%', animation: 'fade-in 0.3s ease-out', marginBottom: '8px' }}>
          <ChatChart chart={chart} />
        </div>
      ))}

      {/* Thinking indicator — shown while waiting for first token */}
      {isPending && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          marginBottom: '16px',
          animation: 'fade-in 0.2s ease-out',
        }}>
          {/* Pulsing dots */}
          <div style={{ display: 'flex', gap: '5px', alignItems: 'center' }}>
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                style={{
                  display: 'inline-block',
                  width: '6px',
                  height: '6px',
                  borderRadius: '50%',
                  background: 'var(--data)',
                  opacity: 0.9,
                  animation: `thinking-dot 1.2s ease-in-out ${i * 0.2}s infinite`,
                }}
              />
            ))}
          </div>
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '11px',
            color: 'var(--ink-muted)',
            letterSpacing: '0.04em',
          }}>
            analyzing…
          </span>
        </div>
      )}

      {/* Streaming text */}
      {streamingText && (
        <div
          style={{
            fontFamily: 'var(--font-display)',
            fontSize: '12.5px',
            lineHeight: 1.6,
            color: 'var(--ink-heavy)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            maxWidth: '78%',
            marginBottom: '16px',
            animation: 'fade-in 0.1s ease-out',
          }}
        >
          {streamingText}
          <span
            style={{
              display: 'inline-block',
              width: '2px',
              height: '15px',
              background: 'var(--data)',
              marginLeft: '2px',
              verticalAlign: 'text-bottom',
              animation: 'cursor-blink 1s step-end infinite',
              borderRadius: '1px',
            }}
          />
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  )
}
