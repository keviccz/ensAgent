import type { ChatMessage } from '@/lib/types'
import { ReasoningBlock } from './ReasoningBlock'
import { ToolCallBlock } from './ToolCallBlock'
import { ChatChart } from './ChatChart'

export function MessageBubble({ message }: { message: ChatMessage }) {
  if (message.role === 'user') {
    return (
      <div
        style={{
          display: 'flex',
          justifyContent: 'flex-end',
          marginBottom: '10px',
          animation: 'fade-up 0.2s ease-out',
        }}
      >
        <div style={{
          maxWidth: '64%',
          background: 'var(--bubble-user)',
          color: 'var(--ink-heavy)',
          borderRadius: '16px 16px 4px 16px',
          padding: '8px 12px',
          fontFamily: 'var(--font-display)',
          fontSize: '12.5px',
          lineHeight: 1.5,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}>
          {message.text}
        </div>
      </div>
    )
  }

  // Assistant message
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        maxWidth: '78%',
        marginBottom: '12px',
        animation: 'fade-up 0.25s ease-out',
      }}
    >
      {message.reasoning && message.reasoning.length > 0 && (
        <ReasoningBlock steps={message.reasoning} />
      )}
      {message.toolCalls?.map((call) => (
        <ToolCallBlock key={call.id} call={call} />
      ))}
      {message.text && (
        <div style={{
          fontFamily: 'var(--font-display)',
          fontSize: '12.5px',
          lineHeight: 1.6,
          color: 'var(--ink-heavy)',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}>
          {message.text}
        </div>
      )}

      {/* Static cluster images — constrained size */}
      {message.images?.map((b64, i) => (
        <div key={i} style={{
          marginTop: '10px',
          maxWidth: '460px',
          borderRadius: '8px',
          overflow: 'hidden',
          border: '1px solid var(--rule-mid)',
          background: 'var(--s1)',
        }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`data:image/png;base64,${b64}`}
            alt={`Clustering plot ${i + 1}`}
            style={{
              display: 'block',
              width: '100%',
              height: 'auto',
              maxHeight: '340px',
              objectFit: 'contain',
              cursor: 'zoom-in',
            }}
            onClick={() => window.open(`data:image/png;base64,${b64}`, '_blank')}
            title="Click to open full size"
          />
          <div style={{
            padding: '4px 10px',
            borderTop: '1px solid var(--rule)',
            fontFamily: 'var(--font-mono)',
            fontSize: '9px',
            color: 'var(--ink-ghost)',
            background: 'var(--s2)',
          }}>
            Click to open full size
          </div>
        </div>
      ))}

      {/* Interactive charts */}
      {message.charts?.map((chart, i) => (
        <ChatChart key={i} chart={chart} />
      ))}
    </div>
  )
}
