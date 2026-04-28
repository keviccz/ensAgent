'use client'
import { useRef, useEffect, useState, useCallback } from 'react'
import type { SpotData, AnnotationResult } from '@/lib/types'
import { getAnnotation } from '@/lib/api'
import { AnnotationPanel } from './AnnotationPanel'

// Scientific layer palette — matches DLPFC cortical layer conventions
const DOMAIN_COLORS: Record<number, string> = {
  1: '#4472C4',  // deep blue   — WM / deep layer
  2: '#ED7D31',  // orange      — L6
  3: '#A9241F',  // brick red   — L5
  4: '#7B5B45',  // brown       — L4
  5: '#D98CB3',  // pink        — L3
  6: '#70AD47',  // olive green — L2
  7: '#4ECDC4',  // teal        — L1
}
const FALLBACK_COLOR = '#9CA3AF'

function domainColor(d: number) { return DOMAIN_COLORS[d] ?? FALLBACK_COLOR }

interface Coords {
  xMin: number; yMin: number; scale: number; offsetX: number; offsetY: number
}

function buildCoords(spots: SpotData[], W: number, H: number, padX = 8, padY = 8): Coords {
  const xs = spots.map(s => s.x)
  const ys = spots.map(s => s.y)
  const xMin = Math.min(...xs), xMax = Math.max(...xs)
  const yMin = Math.min(...ys), yMax = Math.max(...ys)
  const dataW = xMax - xMin || 1
  const dataH = yMax - yMin || 1
  const scale = Math.min((W - padX * 2) / dataW, (H - padY * 2) / dataH)
  const offsetX = padX + ((W - padX * 2) - dataW * scale) / 2
  const offsetY = padY + ((H - padY * 2) - dataH * scale) / 2
  return { xMin, yMin, scale, offsetX, offsetY }
}

function toCanvas(spot: SpotData, c: Coords): [number, number] {
  return [
    c.offsetX + (spot.x - c.xMin) * c.scale,
    c.offsetY + (spot.y - c.yMin) * c.scale,
  ]
}

function drawSpots(
  ctx: CanvasRenderingContext2D,
  spots: SpotData[],
  coords: Coords,
  selectedCluster: number | null,
  W: number,
  H: number,
) {
  ctx.clearRect(0, 0, W, H)

  // Radius: scale to data density — aim for ~55 µm spot representation
  const R = Math.max(1.8, Math.min(3.5, coords.scale * 40))

  // Draw spots — two passes: dim first, selected on top
  const hasSel = selectedCluster !== null

  for (const spot of spots) {
    if (hasSel && spot.cluster === selectedCluster) continue
    const [sx, sy] = toCanvas(spot, coords)
    ctx.beginPath()
    ctx.arc(sx, sy, R, 0, Math.PI * 2)
    ctx.fillStyle = domainColor(spot.cluster)
    ctx.globalAlpha = hasSel ? 0.10 : 0.82
    ctx.fill()
  }

  if (hasSel) {
    for (const spot of spots) {
      if (spot.cluster !== selectedCluster) continue
      const [sx, sy] = toCanvas(spot, coords)
      ctx.beginPath()
      ctx.arc(sx, sy, R + 0.3, 0, Math.PI * 2)
      ctx.fillStyle = domainColor(spot.cluster)
      ctx.globalAlpha = 0.92
      ctx.fill()
    }
  }
  ctx.globalAlpha = 1

  // Legend row at bottom
  const clusterSet: Record<number, true> = {}
  for (const s of spots) clusterSet[s.cluster] = true
  const clusters = Object.keys(clusterSet).map(Number).sort((a, b) => a - b)
  const LY = H - 7
  let lx = 10
  ctx.font = '600 9px "SF Mono","Fira Code",monospace'
  for (const cl of clusters) {
    const color = domainColor(cl)
    const isActive = !hasSel || cl === selectedCluster
    ctx.beginPath()
    ctx.arc(lx + 4, LY - 4, 4, 0, Math.PI * 2)
    ctx.fillStyle = color
    ctx.globalAlpha = isActive ? 1 : 0.3
    ctx.fill()
    ctx.globalAlpha = isActive ? 0.8 : 0.2
    ctx.fillStyle = '#555'
    ctx.fillText(`D${cl}`, lx + 11, LY)
    ctx.globalAlpha = 1
    lx += 30
  }
}

interface Props { spots: SpotData[]; sampleId: string }

export function DomainScatter({ spots, sampleId }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const coordsRef = useRef<Coords | null>(null)
  const spotsRef = useRef<SpotData[]>(spots)
  spotsRef.current = spots

  const [annotation, setAnnotation] = useState<AnnotationResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null)
  // Fade-in: only animate on first draw, instant on re-draws (cluster selection)
  const [visible, setVisible] = useState(false)
  const firstDrawRef = useRef(false)

  // Extracted draw — called both from data/selection change and from ResizeObserver
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
      const coords = buildCoords(spotsRef.current, W, H - 18)
      coordsRef.current = coords
      drawSpots(ctx, spotsRef.current, coords, selectedCluster, W, H)
      if (!firstDrawRef.current) {
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
  }, [spots, selectedCluster])

  const handleClick = useCallback(async (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container || spotsRef.current.length === 0) return

    const rect = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    // Convert to CSS pixels (canvas.getBoundingClientRect already in CSS px)
    const clickX = e.clientX - rect.left
    const clickY = e.clientY - rect.top

    const W = container.clientWidth
    const coords = coordsRef.current ?? buildCoords(spotsRef.current, W, container.clientHeight - 18)

    // Find nearest spot within 16px
    let best: SpotData | null = null
    let bestDist = 16

    for (const spot of spotsRef.current) {
      const [sx, sy] = toCanvas(spot, coords)
      const d = Math.hypot(clickX - sx, clickY - sy)
      if (d < bestDist) { bestDist = d; best = spot }
    }

    if (!best) return
    if (best.cluster === selectedCluster) {
      setSelectedCluster(null)
      setAnnotation(null)
      return
    }

    setSelectedCluster(best.cluster)
    setLoading(true)
    try {
      const result = await getAnnotation(sampleId, best.cluster) as AnnotationResult
      setAnnotation(result)
    } catch (err) {
      console.error('Annotation fetch:', err)
    } finally {
      setLoading(false)
    }
  }, [selectedCluster, sampleId])

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: '12px' }}>

      {/* ── Canvas scatter ─────────────────────────────────── */}
      <div style={{
        background: 'var(--s1)',
        border: '1px solid var(--rule)',
        borderRadius: '12px',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
      }}>
        <div style={{
          padding: '10px 14px',
          borderBottom: '1px solid var(--rule)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexShrink: 0,
        }}>
          <span style={{ fontFamily: 'var(--font-display)', fontSize: '12px', fontWeight: 600, color: 'var(--ink-heavy)' }}>
            Domain Clustering
          </span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            {selectedCluster !== null && (
              <span style={{
                background: domainColor(selectedCluster) + '22',
                color: domainColor(selectedCluster),
                fontFamily: 'var(--font-mono)',
                fontSize: '9.5px',
                fontWeight: 600,
                padding: '1px 7px',
                borderRadius: '5px',
                border: `1px solid ${domainColor(selectedCluster)}44`,
              }}>
                D{selectedCluster}
              </span>
            )}
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', color: 'var(--ink-ghost)' }}>
              {spots.length.toLocaleString()} spots
            </span>
          </div>
        </div>

        <div ref={containerRef} style={{
          flex: 1, position: 'relative', minHeight: '380px',
          opacity: visible ? 1 : 0,
          transition: 'opacity 0.5s ease-out',
        }}>
          {spots.length === 0 ? (
            <div style={{
              position: 'absolute', inset: 0,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10.5px', color: 'var(--ink-ghost)' }}>
                No spatial data loaded
              </span>
            </div>
          ) : (
            <canvas
              ref={canvasRef}
              onClick={handleClick}
              style={{
                position: 'absolute', inset: 0,
                width: '100%', height: '100%',
                cursor: 'crosshair',
                display: 'block',
              }}
            />
          )}
        </div>
      </div>

      {/* ── Annotation panel ───────────────────────────────── */}
      <div style={{
        background: 'var(--s1)',
        border: '1px solid var(--rule)',
        borderRadius: '12px',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
      }}>
        <div style={{
          padding: '10px 14px',
          borderBottom: '1px solid var(--rule)',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          flexShrink: 0,
        }}>
          <span style={{ fontFamily: 'var(--font-display)', fontSize: '12px', fontWeight: 600, color: 'var(--ink-heavy)' }}>
            Annotation
          </span>
          {selectedCluster !== null && (
            <span style={{
              background: 'var(--data-dim)',
              color: 'var(--data)',
              fontFamily: 'var(--font-mono)',
              fontSize: '9.5px',
              padding: '1px 7px',
              borderRadius: '5px',
            }}>
              Domain {selectedCluster}
            </span>
          )}
          {loading && (
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9.5px', color: 'var(--data)', marginLeft: 'auto' }}>
              loading…
            </span>
          )}
        </div>
        <div style={{ flex: 1, overflow: 'auto' }}>
          <AnnotationPanel annotation={annotation} loading={loading} />
        </div>
      </div>
    </div>
  )
}
