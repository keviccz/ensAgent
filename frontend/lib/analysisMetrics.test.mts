import test from 'node:test'
import assert from 'node:assert/strict'

import { computeAnalysisMetrics, buildKpiItems } from './analysisMetrics.ts'

test('computeAnalysisMetrics derives spot, domain, method, and consensus metrics', () => {
  const spots = [
    { spotId: 's1', x: 0, y: 0, cluster: 1 },
    { spotId: 's2', x: 1, y: 1, cluster: 2 },
    { spotId: 's3', x: 2, y: 2, cluster: 2 },
  ]
  const scoreRows = [
    { method: 'IRIS', scores: { '1': 0.9, '2': 0.6 } },
    { method: 'BASS', scores: { '1': 0.8, '2': 0.4 } },
  ]

  const metrics = computeAnalysisMetrics(spots, scoreRows)

  assert.equal(metrics.totalSpots, 3)
  assert.equal(metrics.domainCount, 2)
  assert.equal(metrics.methodCount, 2)
  assert.ok(metrics.meanConsensus !== null)
  assert.ok(Math.abs(metrics.meanConsensus - 0.675) < 1e-9)
})

test('buildKpiItems surfaces dynamic values and handles missing scores', () => {
  const spots = [
    { spotId: 's1', x: 0, y: 0, cluster: 1 },
    { spotId: 's2', x: 1, y: 1, cluster: 3 },
  ]

  const items = buildKpiItems(spots, [])

  assert.deepEqual(
    items.map((item) => item.label),
    ['Total Spots', 'Domains', 'Methods', 'Mean Consensus'],
  )
  assert.equal(items[0].value, '2')
  assert.equal(items[1].value, '2')
  assert.equal(items[2].value, '0')
  assert.equal(items[3].value, '--')
})
