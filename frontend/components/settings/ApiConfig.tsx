'use client'
import { useState } from 'react'
import { CheckCircle, XCircle, Loader, Eye, EyeOff } from 'lucide-react'
import { testConnection } from '@/lib/api'
import type { ApiConfig } from '@/lib/types'

const PROVIDERS = [
  'openai', 'azure', 'anthropic', 'gemini', 'openrouter',
  'deepseek', 'groq', 'cohere', 'mistral', 'together',
  'perplexity', 'fireworks',
]

const PROVIDERS_NEEDING_ENDPOINT = new Set([
  'azure', 'openrouter', 'deepseek', 'groq', 'together', 'fireworks', 'perplexity',
])

function FieldLabel({ children }: { children: React.ReactNode }) {
  return (
    <label style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', letterSpacing: '0.07em', textTransform: 'uppercase', color: 'var(--ink-ghost)', display: 'block', marginBottom: '5px' }}>
      {children}
    </label>
  )
}

function EndpointField({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const [show, setShow] = useState(false)
  return (
    <div>
      <FieldLabel>Endpoint</FieldLabel>
      <div style={{ position: 'relative' }}>
        <input
          type={show ? 'text' : 'password'}
          className="input-base"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="https://…"
          autoComplete="off"
          style={{ paddingRight: '32px' }}
        />
        <button
          type="button"
          onClick={() => setShow((s) => !s)}
          style={{
            position: 'absolute', right: '8px', top: '50%', transform: 'translateY(-50%)',
            background: 'none', border: 'none', cursor: 'pointer',
            color: 'var(--ink-ghost)', padding: '2px', display: 'flex', alignItems: 'center',
          }}
          tabIndex={-1}
        >
          {show ? <EyeOff size={13} /> : <Eye size={13} />}
        </button>
      </div>
    </div>
  )
}

interface Props {
  config: Partial<ApiConfig>
  onChange: (key: keyof ApiConfig, value: string) => void
}

export function ApiConfigPanel({ config, onChange }: Props) {
  const [testState, setTestState] = useState<'idle' | 'loading' | 'ok' | 'fail'>('idle')
  const [testMsg, setTestMsg] = useState('')

  const provider = config.apiProvider ?? ''
  const showEndpoint = PROVIDERS_NEEDING_ENDPOINT.has(provider)
  const showVersion  = provider === 'azure'

  const handleTest = async () => {
    setTestState('loading')
    setTestMsg('')
    try {
      const res = await testConnection({
        api_provider: provider,
        api_key: config.apiKey ?? '',
        api_model: config.apiModel ?? '',
        api_endpoint: config.apiEndpoint ?? '',
        api_version: config.apiVersion ?? '',
      })
      setTestState(res.ok ? 'ok' : 'fail')
      setTestMsg(res.message)
    } catch (err: unknown) {
      setTestState('fail')
      setTestMsg(err instanceof Error ? err.message : 'Connection failed')
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      {/* Provider */}
      <div>
        <FieldLabel>Provider</FieldLabel>
        <select
          value={provider}
          onChange={(e) => onChange('apiProvider', e.target.value)}
          className="input-base"
          style={{ paddingRight: '8px' }}
        >
          <option value="">Select provider…</option>
          {PROVIDERS.map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
      </div>

      {/* API Key */}
      <div>
        <FieldLabel>API Key</FieldLabel>
        <input
          type="password"
          className="input-base"
          value={config.apiKey ?? ''}
          onChange={(e) => onChange('apiKey', e.target.value)}
          placeholder="sk-…"
          autoComplete="off"
        />
      </div>

      {/* Model */}
      <div>
        <FieldLabel>Model</FieldLabel>
        <input
          type="text"
          className="input-base"
          value={config.apiModel ?? ''}
          onChange={(e) => onChange('apiModel', e.target.value)}
          placeholder="e.g. gpt-4o, claude-opus-4-5-20251001"
        />
      </div>

      {/* Endpoint */}
      {showEndpoint && (
        <EndpointField
          value={config.apiEndpoint ?? ''}
          onChange={(v) => onChange('apiEndpoint', v)}
        />
      )}

      {/* API Version (Azure) */}
      {showVersion && (
        <div>
          <FieldLabel>API Version</FieldLabel>
          <input
            type="text"
            className="input-base"
            value={config.apiVersion ?? ''}
            onChange={(e) => onChange('apiVersion', e.target.value)}
            placeholder="2024-12-01-preview"
          />
        </div>
      )}

      {/* Test button + result */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <button
          className="btn-primary"
          onClick={handleTest}
          disabled={testState === 'loading' || !provider || !config.apiKey}
          style={{ fontSize: '12px' }}
        >
          {testState === 'loading' ? (
            <Loader size={12} style={{ animation: 'spin 1s linear infinite' }} />
          ) : null}
          {testState === 'loading' ? 'Testing…' : 'Test Connection'}
        </button>

        {testState === 'ok' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <CheckCircle size={13} style={{ color: 'var(--ok)' }} />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10.5px', color: 'var(--ok)' }}>
              {testMsg || 'Connected'}
            </span>
          </div>
        )}
        {testState === 'fail' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <XCircle size={13} style={{ color: 'var(--err)' }} />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10.5px', color: 'var(--err)', maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {testMsg || 'Failed'}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
