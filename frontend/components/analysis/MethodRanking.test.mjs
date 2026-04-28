import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const source = readFileSync(join(here, 'MethodRanking.tsx'), 'utf8')

test('MethodRanking disables YAxis tick skipping so every method label is visible', () => {
  assert.match(source, /<YAxis[\s\S]*dataKey="method"[\s\S]*interval=\{0\}/)
  assert.match(source, /tick=\{renderMethodTick\}/)
})

test('MethodRanking renders BayesSpace with a smaller y-axis label', () => {
  assert.match(source, /const METHOD_TICK_FONT_SIZE = 10/)
  assert.match(source, /const BAYES_SPACE_TICK_FONT_SIZE = 9/)
  assert.match(source, /method === 'BayesSpace'\s*\?\s*BAYES_SPACE_TICK_FONT_SIZE\s*:\s*METHOD_TICK_FONT_SIZE/)
})
