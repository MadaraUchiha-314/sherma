import { useState } from 'react';
import type { SetStateArgs } from '../../types';

interface Props {
  args: SetStateArgs;
  onChange: (args: SetStateArgs) => void;
}

export function SetStateForm({ args, onChange }: Props) {
  const [newKey, setNewKey] = useState('');
  const [newValue, setNewValue] = useState('');
  const values = args.values || {};

  return (
    <div className="space-y-3">
      <span className="text-xs text-gray-600 font-medium">Values</span>
      <div className="space-y-1">
        {Object.entries(values).map(([key, val]) => (
          <div key={key} className="flex items-center gap-1">
            <input
              className="w-1/3 px-2 py-1 border border-gray-300 rounded text-xs font-mono"
              value={key}
              readOnly
            />
            <input
              className="flex-1 px-2 py-1 border border-gray-300 rounded text-xs font-mono"
              value={val}
              onChange={(e) => {
                const newValues = { ...values, [key]: e.target.value };
                onChange({ ...args, values: newValues });
              }}
            />
            <button
              onClick={() => {
                const newValues = { ...values };
                delete newValues[key];
                onChange({ ...args, values: newValues });
              }}
              className="text-xs text-red-500"
            >
              x
            </button>
          </div>
        ))}
      </div>
      <div className="flex items-center gap-1">
        <input
          className="w-1/3 px-2 py-1 border border-gray-300 rounded text-xs"
          placeholder="key"
          value={newKey}
          onChange={(e) => setNewKey(e.target.value)}
        />
        <input
          className="flex-1 px-2 py-1 border border-gray-300 rounded text-xs"
          placeholder="value"
          value={newValue}
          onChange={(e) => setNewValue(e.target.value)}
        />
        <button
          onClick={() => {
            if (newKey) {
              onChange({ ...args, values: { ...values, [newKey]: newValue } });
              setNewKey('');
              setNewValue('');
            }
          }}
          className="text-xs text-blue-600"
        >
          +
        </button>
      </div>
    </div>
  );
}
