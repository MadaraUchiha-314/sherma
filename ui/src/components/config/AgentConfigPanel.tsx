import { useStore } from '../../store';
import { StateSchemaEditor } from './StateSchemaEditor';

export function AgentConfigPanel() {
  const config = useStore((s) => s.config);
  const activeAgent = useStore((s) => s.activeAgent);
  const updateEntryPoint = useStore((s) => s.updateEntryPoint);
  const updateLangGraphConfig = useStore((s) => s.updateLangGraphConfig);

  const agent = config.agents[activeAgent];
  if (!agent) return null;

  const lgConfig = agent.langgraph_config;

  return (
    <div className="h-full flex flex-col">
      <div className="p-3 border-b border-gray-200">
        <span className="font-semibold text-sm">Agent Config: {activeAgent}</span>
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        <label className="block text-xs">
          <span className="text-gray-600 font-medium">Entry Point</span>
          <select
            className="mt-0.5 w-full px-2 py-1 border border-gray-300 rounded text-xs"
            value={agent.graph.entry_point}
            onChange={(e) => updateEntryPoint(e.target.value)}
          >
            <option value="">Select...</option>
            {agent.graph.nodes.map((n) => (
              <option key={n.name} value={n.name}>{n.name}</option>
            ))}
          </select>
        </label>

        <label className="block text-xs">
          <span className="text-gray-600 font-medium">Recursion Limit</span>
          <input
            className="mt-0.5 w-full px-2 py-1 border border-gray-300 rounded text-xs"
            type="number"
            value={lgConfig?.recursion_limit ?? ''}
            onChange={(e) => {
              const val = e.target.value ? parseInt(e.target.value) : undefined;
              updateLangGraphConfig({ ...lgConfig, recursion_limit: val });
            }}
          />
        </label>

        <label className="block text-xs">
          <span className="text-gray-600 font-medium">Max Concurrency</span>
          <input
            className="mt-0.5 w-full px-2 py-1 border border-gray-300 rounded text-xs"
            type="number"
            value={lgConfig?.max_concurrency ?? ''}
            onChange={(e) => {
              const val = e.target.value ? parseInt(e.target.value) : undefined;
              updateLangGraphConfig({ ...lgConfig, max_concurrency: val });
            }}
          />
        </label>

        <StateSchemaEditor />
      </div>
    </div>
  );
}
