import { useState } from 'react';
import { useStore } from '../../store';
import type { EdgeDef, BranchDef } from '../../types';

export function EdgeDetailPanel() {
  const selectedEdgeId = useStore((s) => s.selectedEdgeId);
  const rfEdges = useStore((s) => s.rfEdges);
  const config = useStore((s) => s.config);
  const activeAgent = useStore((s) => s.activeAgent);
  const updateEdge = useStore((s) => s.updateEdge);
  const deleteEdge = useStore((s) => s.deleteEdge);

  if (!selectedEdgeId) return null;

  const rfEdge = rfEdges.find((e) => e.id === selectedEdgeId);
  if (!rfEdge || rfEdge.data?.edgeIndex === undefined) return null;

  const edgeIndex = rfEdge.data.edgeIndex as number;
  const agent = config.agents[activeAgent];
  if (!agent) return null;

  const edgeDef = agent.graph.edges[edgeIndex];
  if (!edgeDef) return null;

  const isConditional = !!edgeDef.branches;

  const setEdgeDef = (newEdge: EdgeDef) => updateEdge(edgeIndex, newEdge);

  const convertToConditional = () => {
    setEdgeDef({
      source: edgeDef.source,
      branches: [],
      default: edgeDef.target || undefined,
    });
  };

  const convertToStatic = () => {
    setEdgeDef({
      source: edgeDef.source,
      target: edgeDef.default || edgeDef.branches?.[0]?.target || '',
    });
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-3 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <span className="font-semibold text-sm">Edge: {edgeDef.source} → ...</span>
          <button
            onClick={() => deleteEdge(edgeIndex)}
            className="text-xs text-red-500 hover:text-red-700"
          >
            Delete
          </button>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        <div className="flex gap-2">
          <button
            onClick={convertToStatic}
            className={`text-xs px-2 py-1 rounded ${!isConditional ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`}
          >
            Static
          </button>
          <button
            onClick={convertToConditional}
            className={`text-xs px-2 py-1 rounded ${isConditional ? 'bg-purple-600 text-white' : 'bg-gray-200 text-gray-700'}`}
          >
            Conditional
          </button>
        </div>

        {!isConditional && (
          <label className="block text-xs">
            <span className="text-gray-600 font-medium">Target</span>
            <select
              className="mt-0.5 w-full px-2 py-1 border border-gray-300 rounded text-xs"
              value={edgeDef.target || ''}
              onChange={(e) => setEdgeDef({ ...edgeDef, target: e.target.value })}
            >
              <option value="">Select target...</option>
              <option value="__end__">__end__</option>
              {agent.graph.nodes.map((n) => (
                <option key={n.name} value={n.name}>{n.name}</option>
              ))}
            </select>
          </label>
        )}

        {isConditional && (
          <BranchEditor
            branches={edgeDef.branches || []}
            defaultTarget={edgeDef.default}
            nodeNames={['__end__', ...agent.graph.nodes.map((n) => n.name)]}
            onChange={(branches, defaultTarget) =>
              setEdgeDef({ ...edgeDef, branches, default: defaultTarget })
            }
          />
        )}
      </div>
    </div>
  );
}

function BranchEditor({
  branches,
  defaultTarget,
  nodeNames,
  onChange,
}: {
  branches: BranchDef[];
  defaultTarget?: string;
  nodeNames: string[];
  onChange: (branches: BranchDef[], defaultTarget?: string) => void;
}) {
  const [newCondition, setNewCondition] = useState('');
  const [newTarget, setNewTarget] = useState('');

  return (
    <div className="space-y-3">
      <span className="text-xs text-gray-600 font-medium">Branches</span>
      {branches.map((b, idx) => (
        <div key={idx} className="space-y-1 p-2 bg-gray-50 rounded border border-gray-200">
          <input
            className="w-full px-2 py-1 border border-gray-300 rounded text-xs font-mono"
            value={b.condition}
            placeholder="CEL condition"
            onChange={(e) => {
              const newBranches = [...branches];
              newBranches[idx] = { ...b, condition: e.target.value };
              onChange(newBranches, defaultTarget);
            }}
          />
          <div className="flex items-center gap-1">
            <select
              className="flex-1 px-2 py-1 border border-gray-300 rounded text-xs"
              value={b.target}
              onChange={(e) => {
                const newBranches = [...branches];
                newBranches[idx] = { ...b, target: e.target.value };
                onChange(newBranches, defaultTarget);
              }}
            >
              <option value="">Target...</option>
              {nodeNames.map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
            <button
              onClick={() => onChange(branches.filter((_, i) => i !== idx), defaultTarget)}
              className="text-xs text-red-500"
            >
              x
            </button>
          </div>
        </div>
      ))}

      <div className="flex items-center gap-1">
        <input
          className="flex-1 px-2 py-1 border border-gray-300 rounded text-xs"
          placeholder="condition"
          value={newCondition}
          onChange={(e) => setNewCondition(e.target.value)}
        />
        <select
          className="w-24 px-2 py-1 border border-gray-300 rounded text-xs"
          value={newTarget}
          onChange={(e) => setNewTarget(e.target.value)}
        >
          <option value="">Target...</option>
          {nodeNames.map((n) => (
            <option key={n} value={n}>{n}</option>
          ))}
        </select>
        <button
          onClick={() => {
            if (newCondition && newTarget) {
              onChange([...branches, { condition: newCondition, target: newTarget }], defaultTarget);
              setNewCondition('');
              setNewTarget('');
            }
          }}
          className="text-xs text-blue-600"
        >
          +
        </button>
      </div>

      <label className="block text-xs">
        <span className="text-gray-600 font-medium">Default target</span>
        <select
          className="mt-0.5 w-full px-2 py-1 border border-gray-300 rounded text-xs"
          value={defaultTarget || ''}
          onChange={(e) => onChange(branches, e.target.value || undefined)}
        >
          <option value="">None</option>
          {nodeNames.map((n) => (
            <option key={n} value={n}>{n}</option>
          ))}
        </select>
      </label>
    </div>
  );
}
