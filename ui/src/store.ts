import { create } from 'zustand';
import {
  type Node,
  type Edge,
  type OnNodesChange,
  type OnEdgesChange,
  type Connection,
  applyNodeChanges,
  applyEdgeChanges,
  MarkerType,
} from '@xyflow/react';
import type {
  DeclarativeConfig,
  AgentDef,
  NodeDef,
  EdgeDef,
  NodeType,
  NodeArgs,
  LLMDef,
  ToolDef,
  PromptDef,
  SkillDef,
  SubAgentDef,
  HookDef,
  CheckpointerDef,
  StateDef,
  LangGraphConfigDef,
} from './types';
import { defaultArgsForType } from './types';

function makeEmptyAgent(): AgentDef {
  return {
    state: { fields: [{ name: 'messages', type: 'list', default: [] }] },
    graph: { entry_point: '', nodes: [], edges: [] },
  };
}

function makeEmptyConfig(): DeclarativeConfig {
  return {
    agents: { 'my-agent': makeEmptyAgent() },
    llms: [],
    tools: [],
    prompts: [],
    skills: [],
    hooks: [],
    sub_agents: [],
  };
}

export interface AppState {
  // Core state
  config: DeclarativeConfig;
  activeAgent: string;
  nodePositions: Record<string, { x: number; y: number }>;

  // React Flow derived state
  rfNodes: Node[];
  rfEdges: Edge[];

  // Selection
  selectedNodeId: string | null;
  selectedEdgeId: string | null;

  // UI
  yamlDrawerOpen: boolean;

  // Actions
  setConfig: (config: DeclarativeConfig) => void;
  setActiveAgent: (name: string) => void;
  selectNode: (id: string | null) => void;
  selectEdge: (id: string | null) => void;
  toggleYamlDrawer: () => void;

  // Node CRUD
  addNode: (name: string, type: NodeType, position: { x: number; y: number }) => void;
  updateNode: (name: string, args: NodeArgs) => void;
  deleteNode: (name: string) => void;

  // Edge CRUD
  addEdge: (edge: EdgeDef) => void;
  updateEdge: (index: number, edge: EdgeDef) => void;
  deleteEdge: (index: number) => void;
  onConnect: (connection: Connection) => void;

  // Agent config
  updateState: (state: StateDef) => void;
  updateEntryPoint: (ep: string) => void;
  updateLangGraphConfig: (cfg: LangGraphConfigDef | undefined) => void;

  // Resource CRUD
  setLLMs: (llms: LLMDef[]) => void;
  setTools: (tools: ToolDef[]) => void;
  setPrompts: (prompts: PromptDef[]) => void;
  setSkills: (skills: SkillDef[]) => void;
  setSubAgents: (subAgents: SubAgentDef[]) => void;
  setHooks: (hooks: HookDef[]) => void;
  setCheckpointer: (cp: CheckpointerDef | undefined) => void;

  // React Flow callbacks
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;

  // Add agent
  addAgent: (name: string) => void;
  deleteAgent: (name: string) => void;
}

function syncFlowFromConfig(
  config: DeclarativeConfig,
  activeAgent: string,
  positions: Record<string, { x: number; y: number }>
): { rfNodes: Node[]; rfEdges: Edge[] } {
  const agent = config.agents[activeAgent];
  if (!agent) return { rfNodes: [], rfEdges: [] };

  const graph = agent.graph;
  const rfNodes: Node[] = [];
  const rfEdges: Edge[] = [];

  // Add __start__ and __end__ pseudo-nodes
  rfNodes.push({
    id: '__start__',
    type: 'startNode',
    position: positions['__start__'] || { x: 0, y: 300 },
    data: { label: '__start__' },
    deletable: false,
  });

  rfNodes.push({
    id: '__end__',
    type: 'endNode',
    position: positions['__end__'] || { x: 1200, y: 300 },
    data: { label: '__end__' },
    deletable: false,
  });

  // Add user-defined nodes
  for (const node of graph.nodes) {
    const pos = positions[node.name] || { x: 300, y: rfNodes.length * 150 };
    rfNodes.push({
      id: node.name,
      type: 'customNode',
      position: pos,
      data: { nodeDef: node },
    });
  }

  // Add entry_point edge from __start__
  if (graph.entry_point) {
    rfEdges.push({
      id: `__start__->${graph.entry_point}`,
      source: '__start__',
      sourceHandle: 'src-b',
      target: graph.entry_point,
      type: 'staticEdge',
      markerEnd: { type: MarkerType.ArrowClosed },
      data: { isEntryPoint: true },
    });
  }

  // Track source handle usage per node to distribute edges across handles
  const SOURCE_HANDLE_IDS = ['src-a', 'src-b', 'src-c'];
  const sourceHandleCount: Record<string, number> = {};
  const getSourceHandle = (sourceId: string): string => {
    const count = sourceHandleCount[sourceId] || 0;
    sourceHandleCount[sourceId] = count + 1;
    return SOURCE_HANDLE_IDS[count % SOURCE_HANDLE_IDS.length];
  };

  // Convert EdgeDefs to RF edges
  graph.edges.forEach((edgeDef, edgeIdx) => {
    if (edgeDef.target && !edgeDef.branches) {
      // Static edge
      rfEdges.push({
        id: `edge-${edgeIdx}-${edgeDef.source}->${edgeDef.target}`,
        source: edgeDef.source,
        sourceHandle: getSourceHandle(edgeDef.source),
        target: edgeDef.target,
        type: 'staticEdge',
        markerEnd: { type: MarkerType.ArrowClosed },
        data: { edgeIndex: edgeIdx },
      });
    }

    if (edgeDef.branches) {
      edgeDef.branches.forEach((branch, branchIdx) => {
        rfEdges.push({
          id: `edge-${edgeIdx}-branch-${branchIdx}-${edgeDef.source}->${branch.target}`,
          source: edgeDef.source,
          sourceHandle: getSourceHandle(edgeDef.source),
          target: branch.target,
          type: 'conditionalEdge',
          label: branch.condition,
          markerEnd: { type: MarkerType.ArrowClosed },
          data: { edgeIndex: edgeIdx, branchIndex: branchIdx, condition: branch.condition },
        });
      });

      if (edgeDef.default) {
        rfEdges.push({
          id: `edge-${edgeIdx}-default-${edgeDef.source}->${edgeDef.default}`,
          source: edgeDef.source,
          sourceHandle: getSourceHandle(edgeDef.source),
          target: edgeDef.default,
          type: 'conditionalEdge',
          label: 'default',
          markerEnd: { type: MarkerType.ArrowClosed },
          data: { edgeIndex: edgeIdx, isDefault: true },
        });
      }
    }
  });

  return { rfNodes, rfEdges };
}

/** Auto-layout nodes via BFS from entry_point */
export function autoLayout(
  graph: { entry_point: string; nodes: NodeDef[]; edges: EdgeDef[] }
): Record<string, { x: number; y: number }> {
  const positions: Record<string, { x: number; y: number }> = {};
  const adjacency: Record<string, string[]> = {};

  for (const edge of graph.edges) {
    if (!adjacency[edge.source]) adjacency[edge.source] = [];
    if (edge.target) adjacency[edge.source].push(edge.target);
    if (edge.branches) {
      for (const b of edge.branches) adjacency[edge.source].push(b.target);
    }
    if (edge.default) adjacency[edge.source].push(edge.default);
  }

  // BFS from entry_point
  const visited = new Set<string>();
  const queue: { name: string; depth: number }[] = [{ name: graph.entry_point, depth: 0 }];
  visited.add(graph.entry_point);
  const depthNodes: Record<number, string[]> = {};

  while (queue.length > 0) {
    const { name, depth } = queue.shift()!;
    if (!depthNodes[depth]) depthNodes[depth] = [];
    depthNodes[depth].push(name);

    for (const neighbor of adjacency[name] || []) {
      if (neighbor !== '__end__' && !visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push({ name: neighbor, depth: depth + 1 });
      }
    }
  }

  // Add any unreachable nodes
  for (const node of graph.nodes) {
    if (!visited.has(node.name)) {
      const maxDepth = Math.max(0, ...Object.keys(depthNodes).map(Number));
      const d = maxDepth + 1;
      if (!depthNodes[d]) depthNodes[d] = [];
      depthNodes[d].push(node.name);
    }
  }

  const maxDepth = Math.max(0, ...Object.keys(depthNodes).map(Number));

  // Position __start__ and __end__
  positions['__start__'] = { x: 0, y: 300 };
  positions['__end__'] = { x: (maxDepth + 2) * 300, y: 300 };

  for (const [depthStr, names] of Object.entries(depthNodes)) {
    const depth = Number(depthStr);
    names.forEach((name, idx) => {
      positions[name] = {
        x: (depth + 1) * 300,
        y: idx * 150 + (names.length === 1 ? 300 : 150),
      };
    });
  }

  return positions;
}

export const useStore = create<AppState>((set, get) => {
  const initialConfig = makeEmptyConfig();
  const activeAgent = Object.keys(initialConfig.agents)[0];
  const initialPositions: Record<string, { x: number; y: number }> = {
    __start__: { x: 0, y: 300 },
    __end__: { x: 600, y: 300 },
  };
  const { rfNodes, rfEdges } = syncFlowFromConfig(initialConfig, activeAgent, initialPositions);

  return {
    config: initialConfig,
    activeAgent,
    nodePositions: initialPositions,
    rfNodes,
    rfEdges,
    selectedNodeId: null,
    selectedEdgeId: null,
    yamlDrawerOpen: false,

    setConfig: (config) => {
      const state = get();
      const activeAgent = Object.keys(config.agents)[0] || state.activeAgent;
      const agent = config.agents[activeAgent];
      const positions = agent ? autoLayout(agent.graph) : state.nodePositions;
      const { rfNodes, rfEdges } = syncFlowFromConfig(config, activeAgent, positions);
      set({ config, activeAgent, nodePositions: positions, rfNodes, rfEdges, selectedNodeId: null, selectedEdgeId: null });
    },

    setActiveAgent: (name) => {
      const state = get();
      const agent = state.config.agents[name];
      const positions = agent ? autoLayout(agent.graph) : {};
      const { rfNodes, rfEdges } = syncFlowFromConfig(state.config, name, positions);
      set({ activeAgent: name, nodePositions: positions, rfNodes, rfEdges, selectedNodeId: null, selectedEdgeId: null });
    },

    selectNode: (id) => set({ selectedNodeId: id, selectedEdgeId: null }),
    selectEdge: (id) => set({ selectedEdgeId: id, selectedNodeId: null }),
    toggleYamlDrawer: () => set((s) => ({ yamlDrawerOpen: !s.yamlDrawerOpen })),

    addNode: (name, type, position) => {
      const state = get();
      const agent = state.config.agents[state.activeAgent];
      if (!agent) return;
      const newNode: NodeDef = { name, type, args: defaultArgsForType(type) };
      const newGraph = { ...agent.graph, nodes: [...agent.graph.nodes, newNode] };
      const newAgent = { ...agent, graph: newGraph };
      const newConfig = { ...state.config, agents: { ...state.config.agents, [state.activeAgent]: newAgent } };
      const newPositions = { ...state.nodePositions, [name]: position };
      const { rfNodes, rfEdges } = syncFlowFromConfig(newConfig, state.activeAgent, newPositions);
      set({ config: newConfig, nodePositions: newPositions, rfNodes, rfEdges });
    },

    updateNode: (name, args) => {
      const state = get();
      const agent = state.config.agents[state.activeAgent];
      if (!agent) return;
      const newNodes = agent.graph.nodes.map((n) => (n.name === name ? { ...n, args } : n));
      const newGraph = { ...agent.graph, nodes: newNodes };
      const newAgent = { ...agent, graph: newGraph };
      const newConfig = { ...state.config, agents: { ...state.config.agents, [state.activeAgent]: newAgent } };
      const { rfNodes, rfEdges } = syncFlowFromConfig(newConfig, state.activeAgent, state.nodePositions);
      set({ config: newConfig, rfNodes, rfEdges });
    },

    deleteNode: (name) => {
      const state = get();
      const agent = state.config.agents[state.activeAgent];
      if (!agent) return;
      const newNodes = agent.graph.nodes.filter((n) => n.name !== name);
      const newEdges = agent.graph.edges.filter(
        (e) => e.source !== name && e.target !== name
      );
      const newGraph = {
        ...agent.graph,
        nodes: newNodes,
        edges: newEdges,
        entry_point: agent.graph.entry_point === name ? '' : agent.graph.entry_point,
      };
      const newAgent = { ...agent, graph: newGraph };
      const newConfig = { ...state.config, agents: { ...state.config.agents, [state.activeAgent]: newAgent } };
      const newPositions = { ...state.nodePositions };
      delete newPositions[name];
      const { rfNodes, rfEdges } = syncFlowFromConfig(newConfig, state.activeAgent, newPositions);
      set({ config: newConfig, nodePositions: newPositions, rfNodes, rfEdges, selectedNodeId: null });
    },

    addEdge: (edge) => {
      const state = get();
      const agent = state.config.agents[state.activeAgent];
      if (!agent) return;
      const newEdges = [...agent.graph.edges, edge];
      const newGraph = { ...agent.graph, edges: newEdges };
      const newAgent = { ...agent, graph: newGraph };
      const newConfig = { ...state.config, agents: { ...state.config.agents, [state.activeAgent]: newAgent } };
      const { rfNodes, rfEdges } = syncFlowFromConfig(newConfig, state.activeAgent, state.nodePositions);
      set({ config: newConfig, rfNodes, rfEdges });
    },

    updateEdge: (index, edge) => {
      const state = get();
      const agent = state.config.agents[state.activeAgent];
      if (!agent) return;
      const newEdges = [...agent.graph.edges];
      newEdges[index] = edge;
      const newGraph = { ...agent.graph, edges: newEdges };
      const newAgent = { ...agent, graph: newGraph };
      const newConfig = { ...state.config, agents: { ...state.config.agents, [state.activeAgent]: newAgent } };
      const { rfNodes, rfEdges } = syncFlowFromConfig(newConfig, state.activeAgent, state.nodePositions);
      set({ config: newConfig, rfNodes, rfEdges });
    },

    deleteEdge: (index) => {
      const state = get();
      const agent = state.config.agents[state.activeAgent];
      if (!agent) return;
      const newEdges = agent.graph.edges.filter((_, i) => i !== index);
      const newGraph = { ...agent.graph, edges: newEdges };
      const newAgent = { ...agent, graph: newGraph };
      const newConfig = { ...state.config, agents: { ...state.config.agents, [state.activeAgent]: newAgent } };
      const { rfNodes, rfEdges } = syncFlowFromConfig(newConfig, state.activeAgent, state.nodePositions);
      set({ config: newConfig, rfNodes, rfEdges, selectedEdgeId: null });
    },

    onConnect: (connection) => {
      const state = get();
      if (!connection.source || !connection.target) return;
      // If connecting from __start__, set entry_point
      if (connection.source === '__start__') {
        const agent = state.config.agents[state.activeAgent];
        if (!agent) return;
        const newGraph = { ...agent.graph, entry_point: connection.target };
        const newAgent = { ...agent, graph: newGraph };
        const newConfig = { ...state.config, agents: { ...state.config.agents, [state.activeAgent]: newAgent } };
        const { rfNodes, rfEdges } = syncFlowFromConfig(newConfig, state.activeAgent, state.nodePositions);
        set({ config: newConfig, rfNodes, rfEdges });
        return;
      }
      state.addEdge({ source: connection.source, target: connection.target });
    },

    updateState: (stateDef) => {
      const state = get();
      const agent = state.config.agents[state.activeAgent];
      if (!agent) return;
      const newAgent = { ...agent, state: stateDef };
      const newConfig = { ...state.config, agents: { ...state.config.agents, [state.activeAgent]: newAgent } };
      set({ config: newConfig });
    },

    updateEntryPoint: (ep) => {
      const state = get();
      const agent = state.config.agents[state.activeAgent];
      if (!agent) return;
      const newGraph = { ...agent.graph, entry_point: ep };
      const newAgent = { ...agent, graph: newGraph };
      const newConfig = { ...state.config, agents: { ...state.config.agents, [state.activeAgent]: newAgent } };
      const { rfNodes, rfEdges } = syncFlowFromConfig(newConfig, state.activeAgent, state.nodePositions);
      set({ config: newConfig, rfNodes, rfEdges });
    },

    updateLangGraphConfig: (cfg) => {
      const state = get();
      const agent = state.config.agents[state.activeAgent];
      if (!agent) return;
      const newAgent = { ...agent, langgraph_config: cfg };
      const newConfig = { ...state.config, agents: { ...state.config.agents, [state.activeAgent]: newAgent } };
      set({ config: newConfig });
    },

    setLLMs: (llms) => set((s) => ({ config: { ...s.config, llms } })),
    setTools: (tools) => set((s) => ({ config: { ...s.config, tools } })),
    setPrompts: (prompts) => set((s) => ({ config: { ...s.config, prompts } })),
    setSkills: (skills) => set((s) => ({ config: { ...s.config, skills } })),
    setSubAgents: (subAgents) => set((s) => ({ config: { ...s.config, sub_agents: subAgents } })),
    setHooks: (hooks) => set((s) => ({ config: { ...s.config, hooks } })),
    setCheckpointer: (cp) => set((s) => ({ config: { ...s.config, checkpointer: cp } })),

    onNodesChange: (changes) => {
      set((state) => {
        const newRfNodes = applyNodeChanges(changes, state.rfNodes);
        // Sync positions back
        const newPositions = { ...state.nodePositions };
        for (const node of newRfNodes) {
          if (node.position) {
            newPositions[node.id] = node.position;
          }
        }
        return { rfNodes: newRfNodes, nodePositions: newPositions };
      });
    },

    onEdgesChange: (changes) => {
      set((state) => ({
        rfEdges: applyEdgeChanges(changes, state.rfEdges),
      }));
    },

    addAgent: (name) => {
      const state = get();
      const newConfig = {
        ...state.config,
        agents: { ...state.config.agents, [name]: makeEmptyAgent() },
      };
      set({ config: newConfig });
    },

    deleteAgent: (name) => {
      const state = get();
      const newAgents = { ...state.config.agents };
      delete newAgents[name];
      const newConfig = { ...state.config, agents: newAgents };
      const newActive = Object.keys(newAgents)[0] || '';
      const agent = newAgents[newActive];
      const positions = agent ? autoLayout(agent.graph) : {};
      const { rfNodes, rfEdges } = syncFlowFromConfig(newConfig, newActive, positions);
      set({ config: newConfig, activeAgent: newActive, nodePositions: positions, rfNodes, rfEdges });
    },
  };
});
