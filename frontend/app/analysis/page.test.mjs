import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const source = readFileSync(join(here, 'page.tsx'), 'utf8')

test('Analysis page includes the Domain Annotation section', () => {
  assert.match(source, /import \{ DomainScatter \} from '@\/components\/analysis\/DomainScatter'/)
  assert.match(source, /Domain Annotation/)
  assert.match(source, /<DomainScatter spots=\{spots\} sampleId=\{sampleId\} \/>/)
})

test('Analysis consensus scoring uses equal-width map and matrix columns', () => {
  assert.match(source, /gridTemplateColumns:\s*'1fr 1fr'/)
  assert.doesNotMatch(source, /gridTemplateColumns:\s*'3fr 2fr'/)
})
