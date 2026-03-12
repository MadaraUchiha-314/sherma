import { useStore } from '../../store';
import type { CallAgentArgs } from '../../types';

interface Props {
  args: CallAgentArgs;
  onChange: (args: CallAgentArgs) => void;
}

export function CallAgentForm({ args, onChange }: Props) {
  const config = useStore((s) => s.config);

  return (
    <div className="space-y-3">
      <label className="block text-xs">
        <span className="text-gray-600 font-medium">Agent</span>
        <select
          className="mt-0.5 w-full px-2 py-1 border border-gray-300 rounded text-xs"
          value={args.agent?.id || ''}
          onChange={(e) => onChange({ ...args, agent: { id: e.target.value, version: '*' } })}
        >
          <option value="">Select sub-agent...</option>
          {config.sub_agents.map((a) => (
            <option key={a.id} value={a.id}>{a.id}</option>
          ))}
        </select>
      </label>

      <label className="block text-xs">
        <span className="text-gray-600 font-medium">Input (CEL expression)</span>
        <textarea
          className="mt-0.5 w-full px-2 py-1 border border-gray-300 rounded text-xs font-mono"
          rows={2}
          value={args.input || ''}
          onChange={(e) => onChange({ ...args, input: e.target.value })}
        />
      </label>
    </div>
  );
}
