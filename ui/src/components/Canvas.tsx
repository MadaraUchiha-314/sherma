import { useCallback, useRef } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useReactFlow,
  ConnectionMode,
  type Node,
  type Edge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useStore } from '../store';
import { nodeTypes } from './nodes/nodeTypes';
import { edgeTypes } from './edges/edgeTypes';
import type { NodeType } from '../types';
import type { IsValidConnection } from '@xyflow/react';

export function Canvas() {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition } = useReactFlow();
  const rfNodes = useStore((s) => s.rfNodes);
  const rfEdges = useStore((s) => s.rfEdges);
  const onNodesChange = useStore((s) => s.onNodesChange);
  const onEdgesChange = useStore((s) => s.onEdgesChange);
  const onConnect = useStore((s) => s.onConnect);
  const selectNode = useStore((s) => s.selectNode);
  const selectEdge = useStore((s) => s.selectEdge);
  const addNode = useStore((s) => s.addNode);
  const deleteNode = useStore((s) => s.deleteNode);
  const deleteEdge = useStore((s) => s.deleteEdge);
  const config = useStore((s) => s.config);
  const activeAgent = useStore((s) => s.activeAgent);

  const onNodeClick = useCallback((_event: React.MouseEvent, node: Node) => {
    if (node.id === '__start__' || node.id === '__end__') {
      selectNode(null);
    } else {
      selectNode(node.id);
    }
  }, [selectNode]);

  const onEdgeClick = useCallback((_event: React.MouseEvent, edge: Edge) => {
    selectEdge(edge.id);
  }, [selectEdge]);

  const isValidConnection: IsValidConnection = useCallback(() => true, []);

  const onPaneClick = useCallback(() => {
    selectNode(null);
    selectEdge(null);
  }, [selectNode, selectEdge]);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const nodeType = event.dataTransfer.getData('application/sherma-node-type') as NodeType;
      if (!nodeType) return;

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const existingNames = new Set(
        config.agents[activeAgent]?.graph.nodes.map((n) => n.name) || []
      );
      let name = nodeType.replace(/_/g, '_');
      let i = 1;
      while (existingNames.has(name)) {
        name = `${nodeType}_${i++}`;
      }

      const newName = prompt('Node name:', name);
      if (!newName) return;

      addNode(newName, nodeType, position);
    },
    [screenToFlowPosition, addNode, config, activeAgent]
  );

  const onKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      if (event.key === 'Delete' || event.key === 'Backspace') {
        const state = useStore.getState();
        if (state.selectedNodeId) {
          deleteNode(state.selectedNodeId);
        }
        if (state.selectedEdgeId) {
          // Find the edge index from the data
          const rfEdge = state.rfEdges.find((e) => e.id === state.selectedEdgeId);
          if (rfEdge?.data?.edgeIndex !== undefined) {
            deleteEdge(rfEdge.data.edgeIndex as number);
          }
        }
      }
    },
    [deleteNode, deleteEdge]
  );

  return (
    <div ref={reactFlowWrapper} className="h-full w-full" onKeyDown={onKeyDown} tabIndex={0}>
      <ReactFlow
        nodes={rfNodes}
        edges={rfEdges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onEdgeClick={onEdgeClick}
        onPaneClick={onPaneClick}
        onDragOver={onDragOver}
        onDrop={onDrop}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        deleteKeyCode={null}
        connectionMode={ConnectionMode.Loose}
        isValidConnection={isValidConnection}
      >
        <Background />
        <Controls />
        <MiniMap
          nodeColor={(node) => {
            if (node.id === '__start__') return '#1f2937';
            if (node.id === '__end__') return '#1f2937';
            return '#6b7280';
          }}
        />
      </ReactFlow>
    </div>
  );
}
