import { useStore } from '../../store';
import type { ToolNodeArgs, RegistryRef } from '../../types';

interface Props {
  args: ToolNodeArgs;
  onChange: (args: ToolNodeArgs) => void;
}

export function ToolNodeForm({ args, onChange }: Props) {
  const config = useStore((s) => s.config);

  return (
    <div className="space-y-3">
      <p className="text-xs text-gray-500">
        Executes tool calls from the last AI message. Optionally restrict to specific tools.
      </p>
      <div>
        <span className="text-xs text-gray-600 font-medium">Tools (optional)</span>
        <div className="mt-1 space-y-1">
          {(args.tools || []).map((tool, idx) => (
            <div key={idx} className="flex items-center gap-1">
              <select
                className="flex-1 px-2 py-1 border border-gray-300 rounded text-xs"
                value={tool.id}
                onChange={(e) => {
                  const newTools = [...(args.tools || [])];
                  newTools[idx] = { ...newTools[idx], id: e.target.value };
                  onChange({ ...args, tools: newTools });
                }}
              >
                <option value="">Select...</option>
                {config.tools.map((t) => (
                  <option key={t.id} value={t.id}>{t.id}</option>
                ))}
              </select>
              <button
                onClick={() => {
                  const newTools = (args.tools || []).filter((_, i) => i !== idx);
                  onChange({ ...args, tools: newTools.length > 0 ? newTools : undefined });
                }}
                className="text-xs text-red-500"
              >
                x
              </button>
            </div>
          ))}
          <button
            onClick={() => {
              const newTools: RegistryRef[] = [...(args.tools || []), { id: '', version: '*' }];
              onChange({ ...args, tools: newTools });
            }}
            className="text-xs text-blue-600 hover:text-blue-800"
          >
            + Add tool
          </button>
        </div>
      </div>
    </div>
  );
}
