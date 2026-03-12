import type { DataTransformArgs } from '../../types';

interface Props {
  args: DataTransformArgs;
  onChange: (args: DataTransformArgs) => void;
}

export function DataTransformForm({ args, onChange }: Props) {
  return (
    <div className="space-y-3">
      <label className="block text-xs">
        <span className="text-gray-600 font-medium">Expression (CEL)</span>
        <textarea
          className="mt-0.5 w-full px-2 py-1 border border-gray-300 rounded text-xs font-mono"
          rows={3}
          value={args.expression || ''}
          onChange={(e) => onChange({ ...args, expression: e.target.value })}
          placeholder='{"key": "expression"}'
        />
      </label>
    </div>
  );
}
