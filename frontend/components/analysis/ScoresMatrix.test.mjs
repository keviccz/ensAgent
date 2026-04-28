import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const source = readFileSync(join(here, 'ScoresMatrix.tsx'), 'utf8')

test('ScoresMatrix stretches to match the sibling Domain Score Map card', () => {
  assert.match(source, /height:\s*'100%'/)
  assert.doesNotMatch(source, /alignSelf:\s*'start'/)
})

test('ScoresMatrix table and rows use the available card height', () => {
  assert.match(source, /const rowHeight = `\$\{100 \/ rows\.length\}%`/)
  assert.match(source, /<table style=\{\{ width:\s*'100%', height:\s*'100%'/)
  assert.match(source, /<tbody style=\{\{ height:\s*'100%'/)
  assert.match(source, /height:\s*rowHeight/)
})
