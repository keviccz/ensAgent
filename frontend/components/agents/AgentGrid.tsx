import type { AgentState } from '@/lib/types'
import { AgentCard } from './AgentCard'

interface Props {
  agents: AgentState[]
  selectedId: string
  onSelect: (id: string) => void
}

export function AgentGrid({ agents, selectedId, onSelect }: Props) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
      {agents.map((agent, i) => (
        <div key={agent.id} style={{ animationDelay: `${i * 0.04}s` }}>
          <AgentCard
            agent={agent}
            selected={selectedId === agent.id}
            onSelect={onSelect}
          />
        </div>
      ))}
    </div>
  )
}
