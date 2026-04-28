export interface AnalysisMetrics {
  totalSpots: number
  domainCount: number
  methodCount: number
  meanConsensus: number | null
}

export interface KpiItem {
  label: string
  value: string
  sub: string
}

interface SpotLike {
  cluster: number
}

interface ScoresRowLike {
  scores: Record<string, number>
}

function toFixedOrPlaceholder(value: number | null, digits = 2): string {
  return value === null ? '--' : value.toFixed(digits)
}

export function computeAnalysisMetrics(spots: SpotLike[], scoreRows: ScoresRowLike[]): AnalysisMetrics {
  const clusters = new Set(spots.map((spot) => spot.cluster))
  const scoreValues = scoreRows.flatMap((row) =>
    Object.values(row.scores).filter((value): value is number => Number.isFinite(value)),
  )

  return {
    totalSpots: spots.length,
    domainCount: clusters.size,
    methodCount: scoreRows.length,
    meanConsensus: scoreValues.length > 0
      ? scoreValues.reduce((sum, value) => sum + value, 0) / scoreValues.length
      : null,
  }
}

export function buildKpiItems(spots: SpotLike[], scoreRows: ScoresRowLike[]): KpiItem[] {
  const metrics = computeAnalysisMetrics(spots, scoreRows)

  return [
    {
      label: 'Total Spots',
      value: metrics.totalSpots.toLocaleString(),
      sub: 'spatial observations',
    },
    {
      label: 'Domains',
      value: metrics.domainCount.toLocaleString(),
      sub: 'unique clusters',
    },
    {
      label: 'Methods',
      value: metrics.methodCount.toLocaleString(),
      sub: 'scoring rows loaded',
    },
    {
      label: 'Mean Consensus',
      value: toFixedOrPlaceholder(metrics.meanConsensus),
      sub: 'average score cell',
    },
  ]
}
