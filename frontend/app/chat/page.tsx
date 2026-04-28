'use client'
import { useCallback, useEffect, useRef } from 'react'
import { Topbar } from '@/components/layout/Topbar'
import { ChatMessages } from '@/components/chat/ChatMessages'
import { ChatInput } from '@/components/chat/ChatInput'
import { PipelineProgress } from '@/components/chat/PipelineProgress'
import { useStore } from '@/lib/store'
import { createChatStream, getPipelineStatus } from '@/lib/api'

export default function ChatPage() {
  const {
    conversations, activeConversationId, streamingMessage, streamingToolCalls, streamingImages, streamingCharts,
    isWaiting, setWaiting,
    addMessage, appendStreamChunk, finalizeStream, newConversation,
    pipeline, addStreamingToolCall, updateStreamingToolCallResult, addStreamingImage, addStreamingChart, setPipeline,
  } = useStore()

  // Cancel function for current stream
  const cancelStreamRef = useRef<(() => void) | null>(null)

  // Ensure active conversation — guard against React 18 Strict Mode double-fire
  const initRef = useRef(false)
  useEffect(() => {
    if (initRef.current) return
    const state = useStore.getState()
    if (!state.activeConversationId && state.conversations.length === 0) {
      initRef.current = true
      newConversation()
    }
  }, [newConversation])

  useEffect(() => {
    let cancelled = false

    const refreshPipeline = () => {
      getPipelineStatus()
        .then((snapshot) => {
          if (!cancelled) setPipeline(snapshot)
        })
        .catch(() => { /* backend may not be running yet */ })
    }

    refreshPipeline()
    const timer = window.setInterval(refreshPipeline, 2000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [setPipeline])

  const activeConv = conversations.find((c) => c.id === activeConversationId)
  const messages = activeConv?.messages ?? []
  const isStreaming = streamingMessage.length > 0 || streamingToolCalls.length > 0
  const isPending = isWaiting && !isStreaming

  const handleStop = useCallback(() => {
    if (cancelStreamRef.current) {
      cancelStreamRef.current()
      cancelStreamRef.current = null
    }
    setWaiting(false)
    if (activeConversationId) finalizeStream(activeConversationId)
  }, [activeConversationId, finalizeStream, setWaiting])

  const handleSend = useCallback((text: string) => {
    if (!activeConversationId) return

    // If a stream is active, abort it first
    if (cancelStreamRef.current) {
      cancelStreamRef.current()
      cancelStreamRef.current = null
      finalizeStream(activeConversationId)
    }

    const userMsg = {
      id: crypto.randomUUID(),
      role: 'user' as const,
      text,
      timestamp: Date.now(),
    }
    addMessage(activeConversationId, userMsg)
    setWaiting(true)

    const history = messages.map((m) => ({ role: m.role, content: m.text }))
    cancelStreamRef.current = createChatStream(
      text,
      history,
      (chunk) => { setWaiting(false); appendStreamChunk(chunk) },
      (tc) => { setWaiting(false); addStreamingToolCall(tc) },
      (id, result) => updateStreamingToolCallResult(id, result),
      () => { cancelStreamRef.current = null; finalizeStream(activeConversationId!) },
      (err) => {
        cancelStreamRef.current = null
        setWaiting(false)
        finalizeStream(activeConversationId!)
        addMessage(activeConversationId!, {
          id: crypto.randomUUID(),
          role: 'assistant',
          text: `⚠ Connection error: ${err.message}\n\nMake sure the FastAPI server is running on localhost:8000 and your API credentials are configured in Settings.`,
          timestamp: Date.now(),
        })
      },
      (b64) => addStreamingImage(b64),
      (chart) => addStreamingChart(chart),
    )
  }, [activeConversationId, messages, addMessage, setWaiting, appendStreamChunk, addStreamingToolCall, updateStreamingToolCallResult, finalizeStream, addStreamingImage])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      <Topbar
        title="Chat"
        subtitle={activeConv?.title && activeConv.title !== 'New conversation' ? activeConv.title : undefined}
        showDot={isStreaming}
      />
      <ChatMessages
        messages={messages}
        streamingText={isStreaming ? streamingMessage : undefined}
        streamingToolCalls={isStreaming ? streamingToolCalls : []}
        streamingImages={streamingImages}
        streamingCharts={streamingCharts}
        isPending={isPending}
      />
      <PipelineProgress stages={pipeline.stages} />
      <ChatInput
        onSend={handleSend}
        onStop={handleStop}
        isStreaming={isStreaming || isPending}
        disabled={false}
      />
    </div>
  )
}
