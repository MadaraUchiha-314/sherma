import { useState } from 'react';
import type { LLMDef, ToolDef, PromptDef, SkillDef, SubAgentDef, HookDef } from '../../types';

function FormShell({ children, onSave, onCancel }: { children: React.ReactNode; onSave: () => void; onCancel: () => void }) {
  return (
    <div className="space-y-2 p-2 border border-blue-200 rounded bg-blue-50">
      {children}
      <div className="flex gap-2">
        <button onClick={onSave} className="text-xs px-2 py-1 bg-blue-600 text-white rounded hover:bg-blue-700">Save</button>
        <button onClick={onCancel} className="text-xs px-2 py-1 bg-gray-200 text-gray-700 rounded hover:bg-gray-300">Cancel</button>
      </div>
    </div>
  );
}

function Input({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <label className="block text-xs">
      <span className="text-gray-600">{label}</span>
      <input
        className="mt-0.5 w-full px-2 py-1 border border-gray-300 rounded text-xs"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

export function LLMForm({ item, onSave, onCancel }: { item: LLMDef | null; onSave: (v: LLMDef) => void; onCancel: () => void }) {
  const [id, setId] = useState(item?.id || '');
  const [version, setVersion] = useState(item?.version || '*');
  const [provider, setProvider] = useState(item?.provider || 'openai');
  const [modelName, setModelName] = useState(item?.model_name || '');
  return (
    <FormShell onSave={() => onSave({ id, version, provider, model_name: modelName })} onCancel={onCancel}>
      <Input label="ID" value={id} onChange={setId} />
      <Input label="Version" value={version} onChange={setVersion} />
      <Input label="Provider" value={provider} onChange={setProvider} />
      <Input label="Model Name" value={modelName} onChange={setModelName} />
    </FormShell>
  );
}

export function ToolForm({ item, onSave, onCancel }: { item: ToolDef | null; onSave: (v: ToolDef) => void; onCancel: () => void }) {
  const [id, setId] = useState(item?.id || '');
  const [version, setVersion] = useState(item?.version || '*');
  const [importPath, setImportPath] = useState(item?.import_path || '');
  const [url, setUrl] = useState(item?.url || '');
  return (
    <FormShell onSave={() => onSave({ id, version, import_path: importPath || undefined, url: url || undefined })} onCancel={onCancel}>
      <Input label="ID" value={id} onChange={setId} />
      <Input label="Version" value={version} onChange={setVersion} />
      <Input label="Import Path" value={importPath} onChange={setImportPath} />
      <Input label="URL" value={url} onChange={setUrl} />
    </FormShell>
  );
}

export function PromptForm({ item, onSave, onCancel }: { item: PromptDef | null; onSave: (v: PromptDef) => void; onCancel: () => void }) {
  const [id, setId] = useState(item?.id || '');
  const [version, setVersion] = useState(item?.version || '*');
  const [instructions, setInstructions] = useState(item?.instructions || '');
  return (
    <FormShell onSave={() => onSave({ id, version, instructions })} onCancel={onCancel}>
      <Input label="ID" value={id} onChange={setId} />
      <Input label="Version" value={version} onChange={setVersion} />
      <label className="block text-xs">
        <span className="text-gray-600">Instructions</span>
        <textarea
          className="mt-0.5 w-full px-2 py-1 border border-gray-300 rounded text-xs"
          rows={3}
          value={instructions}
          onChange={(e) => setInstructions(e.target.value)}
        />
      </label>
    </FormShell>
  );
}

export function SkillForm({ item, onSave, onCancel }: { item: SkillDef | null; onSave: (v: SkillDef) => void; onCancel: () => void }) {
  const [id, setId] = useState(item?.id || '');
  const [version, setVersion] = useState(item?.version || '*');
  const [url, setUrl] = useState(item?.url || '');
  const [skillCardPath, setSkillCardPath] = useState(item?.skill_card_path || '');
  return (
    <FormShell onSave={() => onSave({ id, version, url: url || undefined, skill_card_path: skillCardPath || undefined })} onCancel={onCancel}>
      <Input label="ID" value={id} onChange={setId} />
      <Input label="Version" value={version} onChange={setVersion} />
      <Input label="URL" value={url} onChange={setUrl} />
      <Input label="Skill Card Path" value={skillCardPath} onChange={setSkillCardPath} />
    </FormShell>
  );
}

export function SubAgentForm({ item, onSave, onCancel }: { item: SubAgentDef | null; onSave: (v: SubAgentDef) => void; onCancel: () => void }) {
  const [id, setId] = useState(item?.id || '');
  const [version, setVersion] = useState(item?.version || '*');
  const [url, setUrl] = useState(item?.url || '');
  const [importPath, setImportPath] = useState(item?.import_path || '');
  const [yamlPath, setYamlPath] = useState(item?.yaml_path || '');
  return (
    <FormShell onSave={() => onSave({ id, version, url: url || undefined, import_path: importPath || undefined, yaml_path: yamlPath || undefined })} onCancel={onCancel}>
      <Input label="ID" value={id} onChange={setId} />
      <Input label="Version" value={version} onChange={setVersion} />
      <Input label="URL" value={url} onChange={setUrl} />
      <Input label="Import Path" value={importPath} onChange={setImportPath} />
      <Input label="YAML Path" value={yamlPath} onChange={setYamlPath} />
    </FormShell>
  );
}

export function HookForm({ item, onSave, onCancel }: { item: HookDef | null; onSave: (v: HookDef) => void; onCancel: () => void }) {
  const [importPath, setImportPath] = useState(item?.import_path || '');
  return (
    <FormShell onSave={() => onSave({ import_path: importPath })} onCancel={onCancel}>
      <Input label="Import Path" value={importPath} onChange={setImportPath} />
    </FormShell>
  );
}
