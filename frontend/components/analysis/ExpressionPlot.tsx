'use client'
import { useRef, useEffect, useState } from 'react'
import type { SpotData, ScoresRow } from '@/lib/types'

/** Compute per-domain consensus score (average across all methods). */
function computeDomainScores(rows: ScoresRow[]): Record<number, number> {
  const sums: Record<number, number[]> = {}
  for (const row of rows) {
    for (const [domain, score] of Object.entries(row.scores)) {
      const d = parseInt(domain)
      if (!sums[d]) sums[d] = []
      sums[d].push(score)
    }
  }
  const result: Record<number, number> = {}
  for (const [d, vals] of Object.entries(sums)) {
    result[parseInt(d)] = vals.reduce((a, b) => a + b, 0) / vals.length
  }
  return result
}

/** Blue (low) → warm orange (high) gradient. t ∈ [0, 1]. */
function scoreColor(t: number): string {
  const r = Math.round(14  + t * (234 - 14))
  const g = Math.round(165 + t * (88  - 165))
  const b = Math.round(233 + t * (36  - 233))
  return `rgb(${r},${g},${b})`
}

interface Props {
  spots: SpotData[]
  scoreRows: ScoresRow[]
}

export function ExpressionPlot({ spots, scoreRows }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)
  const firstDrawRef = useRef(false)
  const spotsRef = useRef(spots)
  const scoreRowsRef = useRef(scoreRows)
  spotsRef.current = spots
  scoreRowsRef.current = scoreRows

  const drawRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    const paint = () => {
      const canvas = canvasRef.current
      const container = containerRef.current
      if (!canvas || !container || spotsRef.current.length === 0) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      const dpr = window.devicePixelRatio || 1
      const W = container.clientWidth
      const H = container.clientHeight
      if (W === 0 || H === 0) return
      canvas.width = W * dpr
      canvas.height = H * dpr
      canvas.style.width = `${W}px`
      canvas.style.height = `${H}px`
      ctx.scale(dpr, dpr)

      const PAD = 8
      const xs = spotsRef.current.map(s => s.x)
      const ys = spotsRef.current.map(s => s.y)
      const xMin = Math.min(...xs), xMax = Math.max(...xs)
      const yMin = Math.min(...ys), yMax = Math.max(...ys)
      const dataW = xMax - xMin || 1
      const dataH = yMax - yMin || 1
      const scale = Math.min((W - PAD * 2) / dataW, (H - PAD * 2) / dataH)
      const offsetX = PAD + ((W - PAD * 2) - dataW * scale) / 2
      const offsetY = PAD + ((H - PAD * 2) - dataH * scale) / 2
      const R = Math.max(1.8, Math.min(3.5, scale * 40))

      ctx.clearRect(0, 0, W, H)

      if (scoreRowsRef.current.length === 0) {
        for (const spot of spotsRef.current) {
          const sx = offsetX + (spot.x - xMin) * scale
          const sy = offsetY + (spot.y - yMin) * scale
          ctx.beginPath()
          ctx.arc(sx, sy, R, 0, Math.PI * 2)
          ctx.fillStyle = '#CBD5E1'
          ctx.globalAlpha = 0.7
          ctx.fill()
        }
        ctx.globalAlpha = 1
        ctx.fillStyle = '#94A3B8'
        ctx.font = '10px monospace'
        ctx.textAlign = 'center'
        ctx.fillText('Score data not loaded', W / 2, H / 2)
        return
      }

      const domainScores = computeDomainScores(scoreRowsRef.current)
      const scoreVals = Object.values(domainScores)
      const minS = Math.min(...scoreVals)
      const maxS = Math.max(...scoreVals, minS + 0.001)

      for (const spot of spotsRef.current) {
        const sx = offsetX + (spot.x - xMin) * scale
        const sy = offsetY + (spot.y - yMin) * scale
        const raw = domainScores[spot.cluster] ?? 0
        const t = (raw - minS) / (maxS - minS)
        ctx.beginPath()
        ctx.arc(sx, sy, R, 0, Math.PI * 2)
        ctx.fillStyle = scoreColor(t)
        ctx.globalAlpha = 0.88
        ctx.fill()
      }
      ctx.globalAlpha = 1

      const clusterXY: Record<number, [number, number, number]> = {}
      for (const spot of spotsRef.current) {
        const sx = offsetX + (spot.x - xMin) * scale
        const sy = offsetY + (spot.y - yMin) * scale
        if (!clusterXY[spot.cluster]) clusterXY[spot.cluster] = [0, 0, 0]
        clusterXY[spot.cluster][0] += sx
        clusterXY[spot.cluster][1] += sy
        clusterXY[spot.cluster][2]++
      }
      ctx.font = '700 9px "SF Mono","Fira Code",monospace'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      for (const [d, [sx, sy, n]] of Object.entries(clusterXY)) {
        const cx = sx / n
        const cy = sy / n
        const score = domainScores[parseInt(d)] ?? 0
        const t = (score - minS) / (maxS - minS)
        ctx.lineWidth = 2.5
        ctx.strokeStyle = 'rgba(255,255,255,0.9)'
        ctx.strokeText(score.toFixed(2), cx, cy)
        ctx.fillStyle = t > 0.5 ? '#7C2D12' : '#1E3A5F'
        ctx.fillText(score.toFixed(2), cx, cy)
      }

      if (!firstDrawRef.current && spotsRef.current.length > 0) {
        firstDrawRef.current = true
        requestAnimationFrame(() => setVisible(true))
      }
    }

    drawRef.current = paint
    paint()

    const container = containerRef.current
    if (!container) return
    const ro = new ResizeObserver(() => { drawRef.current?.() })
    ro.observe(container)
    return () => ro.disconnect()
  }, [spots, scoreRows])

  const hasScores = scoreRows.length > 0

  return (
    <div style={{
      background: 'var(--s1)',
      border: '1px solid var(--rule)',
      borderRadius: '12px',
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* Header */}
      <div style={{
        padding: '10px 14px',
        borderBottom: '1px solid var(--rule)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexShrink: 0,
      }}>
        <span style={{ fontFamily: 'var(--font-display)', fontSize: '12px', fontWeight: 600, color: 'var(--ink-heavy)' }}>
          Domain Score Map
        </span>
        {/* Gradient legend */}
        {hasScores && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--ink-ghost)' }}>low</span>
            <div style={{
              width: '52px', height: '6px', borderRadius: '3px',
              background: 'linear-gradient(to right, #0EA5E9, #EA5824)',
            }} />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--ink-ghost)' }}>high</span>
          </div>
        )}
        {!hasScores && (
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', color: 'var(--ink-ghost)' }}>
            awaiting scores…
          </span>
        )}
      </div>

      {/* Canvas */}
      <div ref={containerRef} style={{
        flex: 1, position: 'relative', minHeight: '340px',
        opacity: visible ? 1 : 0,
        transition: 'opacity 0.5s ease-out 0.1s',
      }}>
        {spots.length === 0 ? (
          <div style={{
            position: 'absolute', inset: 0,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10.5px', color: 'var(--ink-ghost)' }}>
              No spatial data
            </span>
          </div>
        ) : (
          <canvas
            ref={canvasRef}
            style={{
              position: 'absolute', inset: 0,
              width: '100%', height: '100%',
              display: 'block',
            }}
          />
        )}
      </div>
    </div>
  )
}
