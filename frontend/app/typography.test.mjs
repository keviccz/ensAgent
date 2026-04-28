import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

test('frontend source restores existing explicit text sizes to 1x', () => {
  assert.match(readFileSync('app/globals.css', 'utf8'), /font-size: 12\.5px;/)
  assert.match(readFileSync('components/layout/Sidebar.tsx', 'utf8'), /fontSize:\s*'13px'/)
  assert.match(readFileSync('components/analysis/KpiStrip.tsx', 'utf8'), /fontSize:\s*'26px'/)
  assert.match(readFileSync('components/analysis/DomainDistribution.tsx', 'utf8'), /fontSize:\s*10/)
  assert.match(readFileSync('components/analysis/ExpressionPlot.tsx', 'utf8'), /ctx\.font = '10px monospace'/)
})

test('frontend source no longer contains selected 1.2x scaled text sizes', () => {
  assert.doesNotMatch(readFileSync('app/globals.css', 'utf8'), /font-size: 15px;/)
  assert.doesNotMatch(readFileSync('components/layout/Sidebar.tsx', 'utf8'), /fontSize:\s*'15\.6px'/)
  assert.doesNotMatch(readFileSync('components/analysis/KpiStrip.tsx', 'utf8'), /fontSize:\s*'31\.2px'/)
  assert.doesNotMatch(readFileSync('components/analysis/ExpressionPlot.tsx', 'utf8'), /ctx\.font = '12px monospace'/)
})
