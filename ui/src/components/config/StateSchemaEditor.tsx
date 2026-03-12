import { useState } from 'react';
import { useStore } from '../../store';
import type { StateFieldDef } from '../../types';

export function StateSchemaEditor() {
  const config = useStore((s) => s.config);
  const activeAgent = useStore((s) => s.activeAgent);
  const updateState = useStore((s) => s.updateState);
  const [newName, setNewName] = useState('');
  const [newType, setNewType] = useState('str');
  const [newDefault, setNewDefault] = useState('');

  const agent = config.agents[activeAgent];
  if (!agent) return null;

  const fields = agent.state.fields;

  const setFields = (newFields: StateFieldDef[]) => {
    updateState({ fields: newFields });
  };

  const addField = () => {
    if (!newName) return;
    let defaultVal: unknown = newDefault;
    try {
      defaultVal = JSON.parse(newDefault);
    } catch {
      // keep as string
    }
    setFields([...fields, { name: newName, type: newType, default: defaultVal }]);
    setNewName('');
    setNewType('str');
    setNewDefault('');
  };

  return (
    <div className="space-y-2">
      <span className="text-xs text-gray-600 font-medium">State Fields</span>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-gray-500">
            <th className="text-left font-medium py-1">Name</th>
            <th className="text-left font-medium py-1">Type</th>
            <th className="text-left font-medium py-1">Default</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {fields.map((f, idx) => (
            <tr key={idx} className="border-t border-gray-100">
              <td className="py-1">{f.name}</td>
              <td className="py-1">{f.type}</td>
              <td className="py-1 font-mono">{JSON.stringify(f.default)}</td>
              <td className="py-1">
                <button
                  onClick={() => setFields(fields.filter((_, i) => i !== idx))}
                  className="text-red-500 hover:text-red-700"
                >
                  x
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="flex items-center gap-1">
        <input
          className="w-1/4 px-2 py-1 border border-gray-300 rounded text-xs"
          placeholder="name"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
        />
        <input
          className="w-1/4 px-2 py-1 border border-gray-300 rounded text-xs"
          placeholder="type"
          value={newType}
          onChange={(e) => setNewType(e.target.value)}
        />
        <input
          className="flex-1 px-2 py-1 border border-gray-300 rounded text-xs"
          placeholder="default"
          value={newDefault}
          onChange={(e) => setNewDefault(e.target.value)}
        />
        <button onClick={addField} className="text-xs text-blue-600">+</button>
      </div>
    </div>
  );
}
