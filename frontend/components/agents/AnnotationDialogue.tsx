'use client'
import { useState } from 'react'
import { ChevronDown, ChevronRight, CheckCircle, AlertCircle, RotateCcw } from 'lucide-react'
import type { DialogueData, RoundLog, ExpertOutput } from '@/lib/types'

// ── Helpers ───────────────────────────────────────────────────────────────────

const EXPERT_LABELS: Record<string, string> = {
  marker: 'Marker', pathway: 'Pathway', spatial: 'Spatial', vlm: 'Visual (VLM)',
}

function scoreColor(s: number) {
  if (s >= 0.75) return 'var(--ok)'
  if (s >= 0.50) return 'var(--warn)'
  return 'var(--err)'
}

function ScoreBadge({ value }: { value: number }) {
  return (
    <span style={{
      fontFamily: 'var(--font-mono)', fontSize: '10px',
      color: scoreColor(value),
      background: `${scoreColor(value)}18`,
      padding: '1px 7px', borderRadius: '4px',
      border: `1px solid ${scoreColor(value)}30`,
    }}>
      {(value * 100).toFixed(0)}
    </span>
  )
}

// ── Expert row ────────────────────────────────────────────────────────────────

function ExpertRow({ name, data }: { name: string; data: ExpertOutput }) {
  const [open, setOpen] = useState(false)
  const label = EXPERT_LABELS[name.toLowerCase()] ?? name

  return (
    <div style={{
      borderRadius: '6px', overflow: 'hidden',
      border: '1px solid var(--rule)',
      background: 'var(--s1)',
    }}>
      <div
        role="button"
        onClick={() => setOpen((s) => !s)}
        style={{
          display: 'flex', alignItems: 'center', gap: '8px',
          padding: '6px 10px', cursor: 'pointer',
          background: open ? 'var(--s2)' : 'transparent',
          transition: 'background 0.15s',
        }}
      >
        {open ? <ChevronDown size={11} color="var(--ink-ghost)" /> : <ChevronRight size={11} color="var(--ink-ghost)" />}
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-mid)', flex: 1 }}>
          {label}
        </span>
        <ScoreBadge value={data.agent_score ?? 0} />
      </div>
      {open && data.reasoning && (
        <div style={{
          padding: '8px 12px 8px 28px',
          fontFamily: 'var(--font-display)', fontSize: '11.5px',
          color: 'var(--ink-mid)', lineHeight: 1.55,
          borderTop: '1px solid var(--rule)',
        }}>
          {data.reasoning}
        </div>
      )}
    </div>
  )
}

// ── Round card ────────────────────────────────────────────────────────────────

function RoundCard({ round }: { round: RoundLog }) {
  const passed = round.gate?.passed
  const hasCritic = !!round.critic
  const experts = round.experts ?? {}

  return (
    <div style={{
      borderRadius: '10px',
      border: `1px solid ${passed ? 'rgba(34,197,94,0.2)' : 'var(--rule)'}`,
      overflow: 'hidden',
      background: 'var(--s1)',
    }}>
      {/* Round header */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '8px',
        padding: '8px 12px',
        background: passed ? 'rgba(34,197,94,0.05)' : 'var(--s2)',
        borderBottom: '1px solid var(--rule)',
      }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--ink-ghost)' }}>
          Round {round.round}
        </span>
        {round.run_dir && (
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--ink-ghost)', marginLeft: 'auto', opacity: 0.6 }}>
            {round.run_dir}
          </span>
        )}
        {passed !== undefined && (
          <span style={{ marginLeft: round.run_dir ? '0' : 'auto', display: 'flex', alignItems: 'center', gap: '4px' }}>
            {passed
              ? <><CheckCircle size={11} color="var(--ok)" /><span style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', color: 'var(--ok)' }}>Passed</span></>
              : <><AlertCircle size={11} color="var(--warn)" /><span style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', color: 'var(--warn)' }}>Pending</span></>
            }
          </span>
        )}
      </div>

      <div style={{ padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {/* Experts */}
        {Object.keys(experts).length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--ink-ghost)', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: '2px' }}>
              Expert Scores
            </div>
            {Object.entries(experts).map(([k, v]) => (
              <ExpertRow key={k} name={k} data={v} />
            ))}
          </div>
        )}

        {/* Proposer annotation */}
        {round.annotation && (
          <div style={{
            background: 'var(--data-dim)',
            borderRadius: '8px',
            padding: '8px 10px',
            border: '1px solid rgba(14,165,233,0.15)',
          }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--data)', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: '6px' }}>
              Proposer Draft
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', marginBottom: '4px' }}>
              <span style={{ fontFamily: 'var(--font-display)', fontSize: '13px', color: 'var(--ink-heavy)', fontWeight: 600 }}>
                {round.annotation.biological_identity ?? '—'}
              </span>
              <ScoreBadge value={round.annotation.biological_identity_conf ?? 0} />
            </div>
            {round.annotation.alternatives && round.annotation.alternatives.length > 0 && (
              <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap', marginBottom: '4px' }}>
                {round.annotation.alternatives.map((alt) => (
                  <span key={alt.label} style={{
                    fontFamily: 'var(--font-mono)', fontSize: '9.5px',
                    color: 'var(--ink-ghost)', background: 'var(--s2)',
                    padding: '1px 6px', borderRadius: '4px', border: '1px solid var(--rule)',
                  }}>
                    {alt.label} ({(alt.conf * 100).toFixed(0)})
                  </span>
                ))}
              </div>
            )}
            {round.annotation.key_evidence && round.annotation.key_evidence.length > 0 && (
              <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                {round.annotation.key_evidence.slice(0, 5).map((ev) => (
                  <span key={ev} style={{
                    fontFamily: 'var(--font-mono)', fontSize: '9px',
                    color: 'var(--ok)', background: 'rgba(34,197,94,0.07)',
                    padding: '1px 6px', borderRadius: '4px',
                  }}>
                    {ev.replace('Marker: ', '')}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Critic feedback */}
        {hasCritic && (
          <div style={{
            background: 'rgba(234,179,8,0.05)',
            borderRadius: '8px',
            padding: '8px 10px',
            border: '1px solid rgba(234,179,8,0.15)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'rgba(234,179,8,0.9)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                Critic
              </div>
              {round.critic!.critic_score !== undefined && (
                <ScoreBadge value={round.critic!.critic_score} />
              )}
              {round.critic!.rerun_request?.rerun && (
                <span style={{ display: 'flex', alignItems: 'center', gap: '3px', marginLeft: 'auto', fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--warn)' }}>
                  <RotateCcw size={9} /> rerun requested
                </span>
              )}
            </div>
            {round.critic!.reasoning && (
              <p style={{ fontFamily: 'var(--font-display)', fontSize: '11.5px', color: 'var(--ink-mid)', lineHeight: 1.5, margin: 0 }}>
                {round.critic!.reasoning}
              </p>
            )}
            {round.critic!.issues && round.critic!.issues.length > 0 && (
              <div style={{ marginTop: '6px', display: 'flex', flexDirection: 'column', gap: '3px' }}>
                {round.critic!.issues.map((iss, i) => (
                  <div key={i} style={{
                    fontFamily: 'var(--font-mono)', fontSize: '10px',
                    color: iss.blocker ? 'var(--err)' : 'var(--warn)',
                    display: 'flex', gap: '6px',
                  }}>
                    <span>{iss.blocker ? '✗' : '!'}</span>
                    <span>{iss.type}{iss.fix_hint ? ` — ${iss.fix_hint}` : ''}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

interface Props {
  data: DialogueData
}

export function AnnotationDialogue({ data }: Props) {
  const [activeDomain, setActiveDomain] = useState<number>(data.domains[0] ?? 0)
  const rounds: RoundLog[] = data.rounds[activeDomain] ?? []

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      {/* Domain selector */}
      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
        {data.domains.map((d) => (
          <button
            key={d}
            onClick={() => setActiveDomain(d)}
            style={{
              fontFamily: 'var(--font-mono)', fontSize: '11px',
              padding: '3px 10px', borderRadius: '6px', cursor: 'pointer',
              border: `1px solid ${activeDomain === d ? 'rgba(14,165,233,0.4)' : 'var(--rule)'}`,
              background: activeDomain === d ? 'var(--data-dim)' : 'var(--s2)',
              color: activeDomain === d ? 'var(--data)' : 'var(--ink-mid)',
              transition: 'all 0.15s',
            }}
          >
            Domain {d}
          </button>
        ))}
      </div>

      {/* Rounds */}
      {rounds.length === 0 ? (
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-ghost)',
          padding: '16px', textAlign: 'center',
          border: '1px dashed var(--rule)', borderRadius: '8px',
        }}>
          No round logs for domain {activeDomain}
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {rounds.map((r) => <RoundCard key={r.round} round={r} />)}
        </div>
      )}
    </div>
  )
}
