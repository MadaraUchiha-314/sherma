import { useStore } from '../../store';
import type { CallLLMArgs, RegistryRef } from '../../types';

interface Props {
  args: CallLLMArgs;
  onChange: (args: CallLLMArgs) => void;
}

export function CallLLMForm({ args, onChange }: Props) {
  const config = useStore((s) => s.config);

  return (
    <div className="space-y-3">
      <label className="block text-xs">
        <span className="text-gray-600 font-medium">LLM</span>
        <select
          className="mt-0.5 w-full px-2 py-1 border border-gray-300 rounded text-xs"
          value={args.llm?.id || ''}
          onChange={(e) => {
            const llm = config.llms.find((l) => l.id === e.target.value);
            onChange({ ...args, llm: { id: e.target.value, version: llm?.version || '*' } });
          }}
        >
          <option value="">Select LLM...</option>
          {config.llms.map((l) => (
            <option key={l.id} value={l.id}>{l.id}</option>
          ))}
        </select>
        {args.llm?.id && !config.llms.find((l) => l.id === args.llm.id) && (
          <span className="text-orange-500 text-xs">Ref: {args.llm.id} (not in registry)</span>
        )}
      </label>

      <label className="block text-xs">
        <span className="text-gray-600 font-medium">Prompt (CEL expression)</span>
        <textarea
          className="mt-0.5 w-full px-2 py-1 border border-gray-300 rounded text-xs font-mono"
          rows={2}
          value={args.prompt || ''}
          onChange={(e) => onChange({ ...args, prompt: e.target.value })}
        />
      </label>

      <div>
        <span className="text-xs text-gray-600 font-medium">Tools</span>
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

      <div className="space-y-1">
        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            checked={args.use_tools_from_registry || false}
            onChange={(e) => onChange({ ...args, use_tools_from_registry: e.target.checked })}
          />
          <span className="text-gray-600">Use tools from registry</span>
        </label>
        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            checked={args.use_tools_from_loaded_skills || false}
            onChange={(e) => onChange({ ...args, use_tools_from_loaded_skills: e.target.checked })}
          />
          <span className="text-gray-600">Use tools from loaded skills</span>
        </label>
        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            checked={args.use_sub_agents_as_tools || false}
            onChange={(e) => onChange({ ...args, use_sub_agents_as_tools: e.target.checked })}
          />
          <span className="text-gray-600">Use sub-agents as tools</span>
        </label>
      </div>
    </div>
  );
}
