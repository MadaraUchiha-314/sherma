import { Handle, Position } from '@xyflow/react';

export function EndNode() {
  return (
    <div className="bg-gray-800 text-white rounded-full px-4 py-2 text-sm font-semibold shadow-md flex items-center gap-2">
      <Handle type="target" position={Position.Left} className="!bg-red-400 !w-3 !h-3" />
      __end__
      <div className="w-2 h-2 bg-red-400 rounded-full" />
    </div>
  );
}
