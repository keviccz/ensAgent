import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const source = readFileSync(join(here, 'Sidebar.tsx'), 'utf8')
test('Sidebar uses the fixed primary navigation width', () => {
  assert.match(source, /style=\{\{ width: 'var\(--sidebar-w\)', flexShrink: 0 \}\}/)
  assert.doesNotMatch(source, /--sidebar-collapsed-w/)
  assert.doesNotMatch(source, /isCollapsed/)
})

test('Sidebar no longer renders a collapse toggle', () => {
  assert.doesNotMatch(source, /PanelLeftClose|PanelLeftOpen/)
  assert.doesNotMatch(source, /Collapse sidebar|Expand sidebar/)
  assert.doesNotMatch(source, /setIsCollapsed/)
})

test('Sidebar keeps navigation and conversation text visible', () => {
  assert.match(source, /\{label\}/)
  assert.match(source, /New conversation/)
  assert.match(source, /v0\.1\.0 · EnsAgent/)
})
