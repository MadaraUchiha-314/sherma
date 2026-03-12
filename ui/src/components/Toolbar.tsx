import { useRef, useState } from 'react';
import { useStore } from '../store';
import { serializeConfig } from '../yaml/serializer';
import { parseConfig } from '../yaml/parser';

export function Toolbar() {
  const config = useStore((s) => s.config);
  const activeAgent = useStore((s) => s.activeAgent);
  const setConfig = useStore((s) => s.setConfig);
  const setActiveAgent = useStore((s) => s.setActiveAgent);
  const toggleYamlDrawer = useStore((s) => s.toggleYamlDrawer);
  const addAgent = useStore((s) => s.addAgent);
  const deleteAgent = useStore((s) => s.deleteAgent);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);

  const agentNames = Object.keys(config.agents);

  const handleImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = parseConfig(text);
      setConfig(parsed);
      setError(null);
    } catch (err) {
      setError(`Import failed: ${err instanceof Error ? err.message : String(err)}`);
    }
    // Reset file input
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleExport = () => {
    const yamlStr = serializeConfig(config);
    const blob = new Blob([yamlStr], { type: 'text/yaml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'agent.yaml';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleAddAgent = () => {
    const name = prompt('Agent name:');
    if (name && !config.agents[name]) {
      addAgent(name);
      setActiveAgent(name);
    }
  };

  return (
    <div className="flex items-center gap-3 px-4 py-2 bg-white border-b border-gray-200">
      <span className="font-bold text-sm text-gray-800">Sherma</span>

      <div className="flex items-center gap-1 ml-4">
        <label className="text-xs text-gray-500">Agent:</label>
        <select
          className="text-xs px-2 py-1 border border-gray-300 rounded"
          value={activeAgent}
          onChange={(e) => setActiveAgent(e.target.value)}
        >
          {agentNames.map((name) => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
        <button onClick={handleAddAgent} className="text-xs text-blue-600 hover:text-blue-800">+</button>
        {agentNames.length > 1 && (
          <button
            onClick={() => {
              if (confirm(`Delete agent "${activeAgent}"?`)) deleteAgent(activeAgent);
            }}
            className="text-xs text-red-500 hover:text-red-700"
          >
            Del
          </button>
        )}
      </div>

      <div className="ml-auto flex items-center gap-2">
        <input
          ref={fileInputRef}
          type="file"
          accept=".yaml,.yml"
          onChange={handleImport}
          className="hidden"
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          className="text-xs px-3 py-1.5 bg-gray-100 hover:bg-gray-200 rounded border border-gray-300 text-gray-700"
        >
          Import YAML
        </button>
        <button
          onClick={handleExport}
          className="text-xs px-3 py-1.5 bg-gray-100 hover:bg-gray-200 rounded border border-gray-300 text-gray-700"
        >
          Export YAML
        </button>
        <button
          onClick={toggleYamlDrawer}
          className="text-xs px-3 py-1.5 bg-blue-600 hover:bg-blue-700 rounded text-white"
        >
          View YAML
        </button>
      </div>

      {error && (
        <div className="absolute top-12 right-4 bg-red-100 text-red-700 text-xs px-3 py-2 rounded shadow">
          {error}
          <button onClick={() => setError(null)} className="ml-2 text-red-500">x</button>
        </div>
      )}
    </div>
  );
}
