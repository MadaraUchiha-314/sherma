import { useState } from 'react';
import { useStore } from '../../store';
import { NodePalette } from './NodePalette';
import { ResourceList } from './ResourceList';
import { LLMForm, ToolForm, PromptForm, SkillForm, SubAgentForm, HookForm } from './ResourceForms';
import type { LLMDef, ToolDef, PromptDef, SkillDef, SubAgentDef, HookDef } from '../../types';

type Tab = 'nodes' | 'prompts' | 'llms' | 'tools' | 'skills' | 'sub_agents' | 'hooks';

const TABS: { key: Tab; label: string }[] = [
  { key: 'nodes', label: 'Nodes' },
  { key: 'prompts', label: 'Prompts' },
  { key: 'llms', label: 'LLMs' },
  { key: 'tools', label: 'Tools' },
  { key: 'skills', label: 'Skills' },
  { key: 'sub_agents', label: 'Sub-agents' },
  { key: 'hooks', label: 'Hooks' },
];

export function ResourceSidebar() {
  const [tab, setTab] = useState<Tab>('nodes');
  const config = useStore((s) => s.config);
  const setLLMs = useStore((s) => s.setLLMs);
  const setTools = useStore((s) => s.setTools);
  const setPrompts = useStore((s) => s.setPrompts);
  const setSkills = useStore((s) => s.setSkills);
  const setSubAgents = useStore((s) => s.setSubAgents);
  const setHooks = useStore((s) => s.setHooks);

  return (
    <div className="h-full flex flex-col bg-gray-50 border-r border-gray-200">
      <div className="flex flex-wrap gap-1 p-2 border-b border-gray-200">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`text-xs px-2 py-1 rounded ${tab === t.key ? 'bg-blue-600 text-white' : 'bg-white text-gray-600 hover:bg-gray-100'}`}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {tab === 'nodes' && <NodePalette />}

        {tab === 'llms' && (
          <ResourceList<LLMDef>
            title="LLMs"
            items={config.llms}
            getId={(i) => i.id}
            getLabel={(i) => `${i.id} (${i.model_name})`}
            onAdd={(item) => setLLMs([...config.llms, item])}
            onUpdate={(idx, item) => {
              const next = [...config.llms];
              next[idx] = item;
              setLLMs(next);
            }}
            onDelete={(idx) => setLLMs(config.llms.filter((_, i) => i !== idx))}
            renderForm={(item, onSave, onCancel) => (
              <LLMForm item={item} onSave={onSave} onCancel={onCancel} />
            )}
          />
        )}

        {tab === 'tools' && (
          <ResourceList<ToolDef>
            title="Tools"
            items={config.tools}
            getId={(i) => i.id}
            getLabel={(i) => i.id}
            onAdd={(item) => setTools([...config.tools, item])}
            onUpdate={(idx, item) => {
              const next = [...config.tools];
              next[idx] = item;
              setTools(next);
            }}
            onDelete={(idx) => setTools(config.tools.filter((_, i) => i !== idx))}
            renderForm={(item, onSave, onCancel) => (
              <ToolForm item={item} onSave={onSave} onCancel={onCancel} />
            )}
          />
        )}

        {tab === 'prompts' && (
          <ResourceList<PromptDef>
            title="Prompts"
            items={config.prompts}
            getId={(i) => i.id}
            getLabel={(i) => i.id}
            onAdd={(item) => setPrompts([...config.prompts, item])}
            onUpdate={(idx, item) => {
              const next = [...config.prompts];
              next[idx] = item;
              setPrompts(next);
            }}
            onDelete={(idx) => setPrompts(config.prompts.filter((_, i) => i !== idx))}
            renderForm={(item, onSave, onCancel) => (
              <PromptForm item={item} onSave={onSave} onCancel={onCancel} />
            )}
          />
        )}

        {tab === 'skills' && (
          <ResourceList<SkillDef>
            title="Skills"
            items={config.skills}
            getId={(i) => i.id}
            getLabel={(i) => i.id}
            onAdd={(item) => setSkills([...config.skills, item])}
            onUpdate={(idx, item) => {
              const next = [...config.skills];
              next[idx] = item;
              setSkills(next);
            }}
            onDelete={(idx) => setSkills(config.skills.filter((_, i) => i !== idx))}
            renderForm={(item, onSave, onCancel) => (
              <SkillForm item={item} onSave={onSave} onCancel={onCancel} />
            )}
          />
        )}

        {tab === 'sub_agents' && (
          <ResourceList<SubAgentDef>
            title="Sub-agents"
            items={config.sub_agents}
            getId={(i) => i.id}
            getLabel={(i) => i.id}
            onAdd={(item) => setSubAgents([...config.sub_agents, item])}
            onUpdate={(idx, item) => {
              const next = [...config.sub_agents];
              next[idx] = item;
              setSubAgents(next);
            }}
            onDelete={(idx) => setSubAgents(config.sub_agents.filter((_, i) => i !== idx))}
            renderForm={(item, onSave, onCancel) => (
              <SubAgentForm item={item} onSave={onSave} onCancel={onCancel} />
            )}
          />
        )}

        {tab === 'hooks' && (
          <ResourceList<HookDef>
            title="Hooks"
            items={config.hooks}
            getId={(i) => i.import_path}
            getLabel={(i) => i.import_path}
            onAdd={(item) => setHooks([...config.hooks, item])}
            onUpdate={(idx, item) => {
              const next = [...config.hooks];
              next[idx] = item;
              setHooks(next);
            }}
            onDelete={(idx) => setHooks(config.hooks.filter((_, i) => i !== idx))}
            renderForm={(item, onSave, onCancel) => (
              <HookForm item={item} onSave={onSave} onCancel={onCancel} />
            )}
          />
        )}
      </div>
    </div>
  );
}
