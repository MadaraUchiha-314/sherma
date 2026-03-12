import { Handle, Position } from '@xyflow/react';

const SOURCE_HANDLES = [
  { id: 'src-a', top: '25%' },
  { id: 'src-b', top: '50%' },
  { id: 'src-c', top: '75%' },
];

export function StartNode() {
  return (
    <div className="bg-gray-800 text-white rounded-full px-4 py-2 text-sm font-semibold shadow-md flex items-center gap-2">
      <div className="w-2 h-2 bg-green-400 rounded-full" />
      __start__
      {SOURCE_HANDLES.map((h) => (
        <Handle
          key={h.id}
          id={h.id}
          type="source"
          position={Position.Right}
          style={{ top: h.top }}
          className="!bg-green-400 !w-2.5 !h-2.5"
        />
      ))}
    </div>
  );
}
