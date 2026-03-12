import { ReactFlowProvider } from '@xyflow/react';
import { useStore } from '../store';
import { Toolbar } from './Toolbar';
import { ResourceSidebar } from './sidebar/ResourceSidebar';
import { Canvas } from './Canvas';
import { NodeDetailPanel } from './detail/NodeDetailPanel';
import { EdgeDetailPanel } from './detail/EdgeDetailPanel';
import { AgentConfigPanel } from './config/AgentConfigPanel';
import { YamlDrawer } from './yaml-view/YamlDrawer';

export function Layout() {
  const selectedNodeId = useStore((s) => s.selectedNodeId);
  const selectedEdgeId = useStore((s) => s.selectedEdgeId);

  const showDetail = selectedNodeId || selectedEdgeId;

  return (
    <ReactFlowProvider>
      <div className="h-screen w-screen flex flex-col bg-gray-100">
        <Toolbar />
        <div className="flex-1 flex overflow-hidden">
          {/* Left sidebar */}
          <div className="w-56 shrink-0">
            <ResourceSidebar />
          </div>

          {/* Canvas */}
          <div className="flex-1">
            <Canvas />
          </div>

          {/* Right panel */}
          <div className="w-72 shrink-0 bg-white border-l border-gray-200 overflow-y-auto">
            {selectedNodeId && <NodeDetailPanel />}
            {selectedEdgeId && !selectedNodeId && <EdgeDetailPanel />}
            {!showDetail && <AgentConfigPanel />}
          </div>
        </div>
      </div>
      <YamlDrawer />
    </ReactFlowProvider>
  );
}
