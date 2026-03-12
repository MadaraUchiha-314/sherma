import type { InterruptArgs } from '../../types';

interface Props {
  args: InterruptArgs;
  onChange: (args: InterruptArgs) => void;
}

export function InterruptForm({ args, onChange }: Props) {
  return (
    <div className="space-y-3">
      <p className="text-xs text-gray-500">
        Pauses execution and sends the last AI message as the response. Optionally specify a fallback value expression.
      </p>
      <label className="block text-xs">
        <span className="text-gray-600 font-medium">Value (CEL expression, optional)</span>
        <input
          className="mt-0.5 w-full px-2 py-1 border border-gray-300 rounded text-xs font-mono"
          value={args.value || ''}
          onChange={(e) => onChange({ ...args, value: e.target.value || undefined })}
        />
      </label>
    </div>
  );
}
