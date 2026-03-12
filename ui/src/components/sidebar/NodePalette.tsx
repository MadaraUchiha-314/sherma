import type { NodeType } from '../../types';
import { NODE_COLORS, NODE_TYPE_LABELS } from '../../types';

const NODE_TYPES: NodeType[] = [
  'call_llm',
  'tool_node',
  'call_agent',
  'data_transform',
  'set_state',
  'interrupt',
];

export function NodePalette() {
  const onDragStart = (event: React.DragEvent, nodeType: NodeType) => {
    event.dataTransfer.setData('application/sherma-node-type', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Nodes</h3>
      {NODE_TYPES.map((type) => (
        <div
          key={type}
          draggable
          onDragStart={(e) => onDragStart(e, type)}
          className="flex items-center gap-2 px-3 py-2 rounded-md border border-gray-200 cursor-grab active:cursor-grabbing hover:border-gray-400 bg-white transition-colors"
        >
          <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: NODE_COLORS[type] }} />
          <span className="text-sm text-gray-700">{NODE_TYPE_LABELS[type]}</span>
        </div>
      ))}
    </div>
  );
}
