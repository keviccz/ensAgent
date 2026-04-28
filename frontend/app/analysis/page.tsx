'use client'
import { useEffect, useState } from 'react'
import { Topbar } from '@/components/layout/Topbar'
import { KpiStrip } from '@/components/analysis/KpiStrip'
import { ExpressionPlot } from '@/components/analysis/ExpressionPlot'
import { ScoresMatrix } from '@/components/analysis/ScoresMatrix'
import { DomainScatter } from '@/components/analysis/DomainScatter'
import { DomainDistribution } from '@/components/analysis/DomainDistribution'
import { MethodRanking } from '@/components/analysis/MethodRanking'
import { getSpatialData, getScores } from '@/lib/api'
import { useStore } from '@/lib/store'
import type { SpotData, ScoresRow } from '@/lib/types'

export default function AnalysisPage() {
  const sampleId = useStore((s) => s.config.sampleId || 'DLPFC_151507')
  const [spots, setSpots] = useState<SpotData[]>([])
  const [scoreRows, setScoreRows] = useState<ScoresRow[]>([])
  const [loading, setLoading] = useState(true)
  const [fetchError, setFetchError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    setFetchError(null)
    Promise.allSettled([
      getSpatialData(sampleId).then((d) => setSpots((d as any).spots ?? [])),
      getScores(sampleId).then((d) => setScoreRows((d as any).rows ?? [])),
    ]).then((results) => {
      const failures = results.filter((r) => r.status === 'rejected')
      if (failures.length > 0) {
        const msg = (failures[0] as PromiseRejectedResult).reason?.message ?? 'Unknown error'
        setFetchError(`Failed to fetch analysis data: ${msg}. Is the FastAPI server running?`)
      }
    }).finally(() => setLoading(false))
  }, [sampleId])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      <Topbar
        title="Spatial Analysis"
        subtitle={sampleId}
        showDot={loading}
      />
      <div style={{ flex: 1, overflowY: 'auto', padding: '20px 24px' }}>
        {fetchError && (
          <div style={{
            background: 'rgba(239,68,68,0.06)',
            border: '1px solid rgba(239,68,68,0.2)',
            borderRadius: '8px',
            padding: '10px 14px',
            marginBottom: '16px',
            fontFamily: 'var(--font-mono)',
            fontSize: '11px',
            color: 'var(--err)',
            lineHeight: 1.5,
          }}>
            {fetchError}
          </div>
        )}
        <KpiStrip spots={spots} scoreRows={scoreRows} />

        {/* Domain annotation */}
        <section style={{ marginBottom: '20px' }}>
          <div className="section-head">Domain Annotation</div>
          <DomainScatter spots={spots} sampleId={sampleId} />
        </section>

        {/* Consensus scoring */}
        <section style={{ marginBottom: '20px' }}>
          <div className="section-head">Consensus Scoring</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', alignItems: 'stretch' }}>
            <ExpressionPlot spots={spots} scoreRows={scoreRows} />
            <ScoresMatrix rows={scoreRows} />
          </div>
        </section>

        {/* Statistics */}
        <section style={{ marginBottom: '20px' }}>
          <div className="section-head">Statistics</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <DomainDistribution spots={spots} />
            <MethodRanking rows={scoreRows} />
          </div>
        </section>
      </div>
    </div>
  )
}
