import { Handle, Position } from '@xyflow/react';
import type { NodeDef } from '../../types';
import { NODE_COLORS, NODE_TYPE_LABELS } from '../../types';

interface BaseNodeProps {
  nodeDef: NodeDef;
  selected?: boolean;
}

const SOURCE_HANDLES = [
  { id: 'src-a', top: '25%' },
  { id: 'src-b', top: '50%' },
  { id: 'src-c', top: '75%' },
];

export function BaseNode({ nodeDef, selected }: BaseNodeProps) {
  const color = NODE_COLORS[nodeDef.type];
  return (
    <div
      className={`bg-white rounded-lg shadow-md min-w-[180px] border-2 ${selected ? 'border-blue-500' : 'border-gray-200'}`}
    >
      <div
        className="px-3 py-1.5 rounded-t-md text-white text-xs font-semibold"
        style={{ backgroundColor: color }}
      >
        {NODE_TYPE_LABELS[nodeDef.type]}
      </div>
      <div className="px-3 py-2 text-sm font-medium text-gray-800">
        {nodeDef.name}
      </div>
      <Handle type="target" position={Position.Left} className="!bg-gray-400 !w-3 !h-3" />
      {SOURCE_HANDLES.map((h) => (
        <Handle
          key={h.id}
          id={h.id}
          type="source"
          position={Position.Right}
          style={{ top: h.top }}
          className="!bg-gray-400 !w-2.5 !h-2.5"
        />
      ))}
    </div>
  );
}
