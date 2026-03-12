import { useStore } from '../../store';
import { NODE_TYPE_LABELS, NODE_COLORS } from '../../types';
import type { NodeDef, CallLLMArgs, ToolNodeArgs, CallAgentArgs, DataTransformArgs, SetStateArgs, InterruptArgs } from '../../types';
import { CallLLMForm } from './CallLLMForm';
import { ToolNodeForm } from './ToolNodeForm';
import { CallAgentForm } from './CallAgentForm';
import { DataTransformForm } from './DataTransformForm';
import { SetStateForm } from './SetStateForm';
import { InterruptForm } from './InterruptForm';
import { serializeNodeYaml } from '../../yaml/serializer';

export function NodeDetailPanel() {
  const selectedNodeId = useStore((s) => s.selectedNodeId);
  const config = useStore((s) => s.config);
  const activeAgent = useStore((s) => s.activeAgent);
  const updateNode = useStore((s) => s.updateNode);
  const deleteNode = useStore((s) => s.deleteNode);

  if (!selectedNodeId) return null;

  const agent = config.agents[activeAgent];
  if (!agent) return null;

  const nodeDef = agent.graph.nodes.find((n) => n.name === selectedNodeId);
  if (!nodeDef) return null;

  const renderForm = (node: NodeDef) => {
    const onChange = (args: NodeDef['args']) => updateNode(node.name, args);

    switch (node.type) {
      case 'call_llm':
        return <CallLLMForm args={node.args as CallLLMArgs} onChange={onChange} />;
      case 'tool_node':
        return <ToolNodeForm args={node.args as ToolNodeArgs} onChange={onChange} />;
      case 'call_agent':
        return <CallAgentForm args={node.args as CallAgentArgs} onChange={onChange} />;
      case 'data_transform':
        return <DataTransformForm args={node.args as DataTransformArgs} onChange={onChange} />;
      case 'set_state':
        return <SetStateForm args={node.args as SetStateArgs} onChange={onChange} />;
      case 'interrupt':
        return <InterruptForm args={node.args as InterruptArgs} onChange={onChange} />;
    }
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-3 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: NODE_COLORS[nodeDef.type] }} />
            <span className="font-semibold text-sm">{nodeDef.name}</span>
          </div>
          <button
            onClick={() => deleteNode(nodeDef.name)}
            className="text-xs text-red-500 hover:text-red-700"
          >
            Delete
          </button>
        </div>
        <span className="text-xs text-gray-500">{NODE_TYPE_LABELS[nodeDef.type]}</span>
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {renderForm(nodeDef)}
      </div>
      <div className="p-3 border-t border-gray-200">
        <details>
          <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">Node YAML</summary>
          <pre className="mt-2 text-xs bg-gray-100 p-2 rounded overflow-auto max-h-[200px] text-gray-700">
            {serializeNodeYaml(nodeDef)}
          </pre>
        </details>
      </div>
    </div>
  );
}
