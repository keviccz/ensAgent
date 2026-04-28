'use client'
import { useRef, useEffect, useState, useCallback } from 'react'
import type { SpotData, AnnotationResult } from '@/lib/types'
import { getSpatialData, getAnnotation } from '@/lib/api'

const DOMAIN_COLORS: Record<number, string> = {
  1: '#4472C4',
  2: '#ED7D31',
  3: '#A9241F',
  4: '#7B5B45',
  5: '#D98CB3',
  6: '#70AD47',
  7: '#4ECDC4',
}
const FALLBACK_COLOR = '#9CA3AF'

function domainColor(d: number) { return DOMAIN_COLORS[d] ?? FALLBACK_COLOR }

interface Coords {
  xMin: number; yMin: number; scale: number; offsetX: number; offsetY: number
}

function buildCoords(spots: SpotData[], W: number, H: number, pad = 8): Coords {
  const xs = spots.map(s => s.x)
  const ys = spots.map(s => s.y)
  const xMin = Math.min(...xs), xMax = Math.max(...xs)
  const yMin = Math.min(...ys), yMax = Math.max(...ys)
  const dataW = xMax - xMin || 1
  const dataH = yMax - yMin || 1
  const scale = Math.min((W - pad * 2) / dataW, (H - pad * 2) / dataH)
  const offsetX = pad + ((W - pad * 2) - dataW * scale) / 2
  const offsetY = pad + ((H - pad * 2) - dataH * scale) / 2
  return { xMin, yMin, scale, offsetX, offsetY }
}

function toCanvas(spot: SpotData, c: Coords): [number, number] {
  return [c.offsetX + (spot.x - c.xMin) * c.scale, c.offsetY + (spot.y - c.yMin) * c.scale]
}

function drawCanvas(
  ctx: CanvasRenderingContext2D,
  spots: SpotData[],
  coords: Coords,
  sel: number | null,
  W: number,
  H: number,
) {
  ctx.clearRect(0, 0, W, H)
  const R = Math.max(1.6, Math.min(3.4, coords.scale * 40))
  const hasSel = sel !== null

  for (const spot of spots) {
    if (hasSel && spot.cluster === sel) continue
    const [sx, sy] = toCanvas(spot, coords)
    ctx.beginPath()
    ctx.arc(sx, sy, R, 0, Math.PI * 2)
    ctx.fillStyle = domainColor(spot.cluster)
    ctx.globalAlpha = hasSel ? 0.10 : 0.82
    ctx.fill()
  }
  if (hasSel) {
    for (const spot of spots) {
      if (spot.cluster !== sel) continue
      const [sx, sy] = toCanvas(spot, coords)
      ctx.beginPath()
      ctx.arc(sx, sy, R + 0.4, 0, Math.PI * 2)
      ctx.fillStyle = domainColor(spot.cluster)
      ctx.globalAlpha = 0.92
      ctx.fill()
    }
  }
  ctx.globalAlpha = 1

  // Legend at bottom
  const clusterSet: Record<number, true> = {}
  for (const s of spots) clusterSet[s.cluster] = true
  const clusters = Object.keys(clusterSet).map(Number).sort((a, b) => a - b)
  const LY = H - 6
  let lx = 10
  ctx.font = '9px "SF Mono","Fira Code",monospace'
  for (const cl of clusters) {
    const active = !hasSel || cl === sel
    ctx.beginPath()
    ctx.arc(lx + 4, LY - 4, 4, 0, Math.PI * 2)
    ctx.fillStyle = domainColor(cl)
    ctx.globalAlpha = active ? 1 : 0.25
    ctx.fill()
    ctx.globalAlpha = active ? 0.8 : 0.2
    ctx.fillStyle = '#888'
    ctx.fillText(`D${cl}`, lx + 11, LY)
    ctx.globalAlpha = 1
    lx += 30
  }
}

// Compute optimal canvas height so tissue fills the container with minimal whitespace
function computeIdealHeight(spots: SpotData[], containerW: number): number {
  if (spots.length === 0) return 200
  const xs = spots.map(s => s.x)
  const ys = spots.map(s => s.y)
  const dataW = Math.max(...xs) - Math.min(...xs) || 1
  const dataH = Math.max(...ys) - Math.min(...ys) || 1
  const ratio = dataH / dataW
  // Use 72% of container width to compute canvas content area
  const canvasDataW = containerW * 0.72
  const idealH = canvasDataW * ratio + 20
  return Math.min(Math.max(idealH, 160), 300)
}

interface Props {
  sampleId: string
}

export function ChatScatter({ sampleId }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const outerRef = useRef<HTMLDivElement>(null)   // for width measurement
  const containerRef = useRef<HTMLDivElement>(null)
  const coordsRef = useRef<Coords | null>(null)
  const spotsRef = useRef<SpotData[]>([])

  const [spots, setSpots] = useState<SpotData[]>([])
  const [loading, setLoading] = useState(true)
  const [containerH, setContainerH] = useState(200)
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null)
  const [annotation, setAnnotation] = useState<AnnotationResult | null>(null)
  const [annLoading, setAnnLoading] = useState(false)

  const hasSpatial = spots.length > 0 && spots.some(s => s.x !== 0 || s.y !== 0)

  useEffect(() => {
    setLoading(true)
    getSpatialData(sampleId)
      .then((data) => {
        const s = (data.spots ?? []) as SpotData[]
        setSpots(s)
        spotsRef.current = s
        // Compute height from measured container width
        const w = outerRef.current?.clientWidth || 440
        setContainerH(computeIdealHeight(s, w))
      })
      .catch(() => { /* ignore */ })
      .finally(() => setLoading(false))
  }, [sampleId])

  // Re-render canvas on spots/selection/height change
  useEffect(() => {
    if (!hasSpatial) return
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container || spots.length === 0) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const dpr = window.devicePixelRatio || 1
    const W = container.clientWidth
    const H = container.clientHeight
    canvas.width = W * dpr
    canvas.height = H * dpr
    canvas.style.width = `${W}px`
    canvas.style.height = `${H}px`
    ctx.scale(dpr, dpr)
    const coords = buildCoords(spots, W, H - 18)
    coordsRef.current = coords
    drawCanvas(ctx, spots, coords, selectedCluster, W, H)
  }, [spots, selectedCluster, hasSpatial, containerH])

  const handleCanvasClick = useCallback(async (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container || spotsRef.current.length === 0) return
    const rect = canvas.getBoundingClientRect()
    const clickX = e.clientX - rect.left
    const clickY = e.clientY - rect.top
    const W = container.clientWidth
    const coords = coordsRef.current ?? buildCoords(spotsRef.current, W, container.clientHeight - 18)
    let best: SpotData | null = null
    let bestDist = 18
    for (const spot of spotsRef.current) {
      const [sx, sy] = toCanvas(spot, coords)
      const d = Math.hypot(clickX - sx, clickY - sy)
      if (d < bestDist) { bestDist = d; best = spot }
    }
    if (!best) return
    if (best.cluster === selectedCluster) { setSelectedCluster(null); setAnnotation(null); return }
    setSelectedCluster(best.cluster)
    setAnnLoading(true)
    try {
      const result = await getAnnotation(sampleId, best.cluster) as AnnotationResult
      setAnnotation(result)
    } catch { /* ignore */ } finally { setAnnLoading(false) }
  }, [selectedCluster, sampleId])

  const handleDomainClick = useCallback(async (cl: number) => {
    if (cl === selectedCluster) { setSelectedCluster(null); setAnnotation(null); return }
    setSelectedCluster(cl)
    setAnnLoading(true)
    try {
      const result = await getAnnotation(sampleId, cl) as AnnotationResult
      setAnnotation(result)
    } catch { /* ignore */ } finally { setAnnLoading(false) }
  }, [selectedCluster, sampleId])

  if (loading) {
    return (
      <div style={{ border: '1px solid var(--rule)', borderRadius: '10px', padding: '24px', textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-ghost)' }}>
        loading spatial data…
      </div>
    )
  }

  if (spots.length === 0) {
    return (
      <div style={{ border: '1px solid var(--rule)', borderRadius: '10px', padding: '24px', textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--ink-ghost)' }}>
        No spatial data found for {sampleId}
      </div>
    )
  }

  const clusterSet2: Record<number, true> = {}
  for (const s of spots) clusterSet2[s.cluster] = true
  const clusters = Object.keys(clusterSet2).map(Number).sort((a, b) => a - b)
  const counts: Record<number, number> = {}
  for (const s of spots) counts[s.cluster] = (counts[s.cluster] || 0) + 1

  return (
    <div ref={outerRef} style={{ border: '1px solid var(--rule)', borderRadius: '10px', overflow: 'hidden', background: 'var(--s1)', maxWidth: '520px' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 12px', borderBottom: '1px solid var(--rule)', background: 'var(--s2)' }}>
        <span style={{ fontFamily: 'var(--font-display)', fontSize: '11.5px', fontWeight: 600, color: 'var(--ink-heavy)' }}>
          Cluster Map — {sampleId}
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {selectedCluster !== null && (
            <span style={{ background: domainColor(selectedCluster) + '22', color: domainColor(selectedCluster), fontFamily: 'var(--font-mono)', fontSize: '9px', fontWeight: 600, padding: '1px 7px', borderRadius: '4px', border: `1px solid ${domainColor(selectedCluster)}44` }}>
              D{selectedCluster}
            </span>
          )}
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--ink-ghost)' }}>
            {spots.length.toLocaleString()} spots · {clusters.length} domains
          </span>
        </div>
      </div>

      {hasSpatial ? (
        <div ref={containerRef} style={{ position: 'relative', height: `${containerH}px`, background: 'var(--s0)' }}>
          <canvas
            ref={canvasRef}
            onClick={handleCanvasClick}
            style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', cursor: 'crosshair', display: 'block' }}
          />
        </div>
      ) : (
        <div style={{ padding: '12px' }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', color: 'var(--ink-ghost)', marginBottom: '8px' }}>
            Click a domain to view annotation
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {clusters.map((cl) => {
              const pct = ((counts[cl] || 0) / spots.length * 100).toFixed(1)
              const isActive = selectedCluster === cl
              return (
                <button key={cl} onClick={() => void handleDomainClick(cl)} style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 10px', borderRadius: '7px', border: `1.5px solid ${isActive ? domainColor(cl) : 'var(--rule)'}`, background: isActive ? domainColor(cl) + '18' : 'var(--s2)', cursor: 'pointer', transition: 'all 0.15s' }}>
                  <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: domainColor(cl), flexShrink: 0 }} />
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', fontWeight: 600, color: 'var(--ink-heavy)' }}>D{cl}</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--ink-ghost)' }}>{counts[cl]} ({pct}%)</span>
                </button>
              )
            })}
          </div>
        </div>
      )}

      {/* Annotation panel */}
      {(selectedCluster !== null || annLoading) && (
        <div style={{ borderTop: '1px solid var(--rule)', padding: '10px 12px', background: 'var(--s0)' }}>
          {annLoading
            ? <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10.5px', color: 'var(--ink-ghost)' }}>loading annotation…</span>
            : annotation ? <CompactAnnotation annotation={annotation} color={domainColor(selectedCluster!)} /> : null
          }
        </div>
      )}
    </div>
  )
}

function CompactAnnotation({ annotation, color }: { annotation: AnnotationResult; color: string }) {
  const hasData = annotation.confidence > 0 || !annotation.label.match(/^(Cluster|Domain) \d+$/)
  if (!hasData) {
    return <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10.5px', color: 'var(--ink-ghost)' }}>{annotation.interpretation || 'Annotation not yet available — run Stage D first.'}</div>
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
        <span style={{ fontFamily: 'var(--font-display)', fontSize: '12.5px', fontWeight: 700, color: 'var(--ink-heavy)' }}>{annotation.label}</span>
        {annotation.confidence > 0 && (
          <span style={{ background: color + '1a', color, fontFamily: 'var(--font-mono)', fontSize: '9px', fontWeight: 600, padding: '1px 7px', borderRadius: '4px', border: `1px solid ${color}33` }}>
            {(annotation.confidence * 100).toFixed(0)}% conf
          </span>
        )}
      </div>
      {annotation.markerGenes.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
          {annotation.markerGenes.slice(0, 6).map((g) => (
            <span key={g} style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', fontStyle: 'italic', padding: '1px 5px', borderRadius: '3px', background: 'var(--s2)', color: 'var(--ink-muted)', border: '1px solid var(--rule)' }}>
              {g.replace(/^Marker:\s*/, '')}
            </span>
          ))}
        </div>
      )}
      {annotation.interpretation && (
        <p style={{ fontFamily: 'var(--font-display)', fontSize: '11px', lineHeight: 1.5, color: 'var(--ink-mid)', margin: 0 }}>
          {annotation.interpretation.slice(0, 220)}{annotation.interpretation.length > 220 ? '…' : ''}
        </p>
      )}
    </div>
  )
}
