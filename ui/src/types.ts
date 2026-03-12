/** TypeScript mirrors of schema.py Pydantic models */

export interface RegistryRef {
  id: string;
  version?: string;
}

export interface StateFieldDef {
  name: string;
  type: string;
  default?: unknown;
}

export interface StateDef {
  fields: StateFieldDef[];
}

export interface ResponseFormatDef {
  name: string;
  description?: string;
  schema: Record<string, unknown>;
}

export interface CallLLMArgs {
  llm: RegistryRef;
  prompt: string;
  tools?: RegistryRef[];
  use_tools_from_registry?: boolean;
  use_tools_from_loaded_skills?: boolean;
  use_sub_agents_as_tools?: boolean;
  response_format?: ResponseFormatDef;
}

export interface ToolNodeArgs {
  tools?: RegistryRef[];
}

export interface CallAgentArgs {
  agent: RegistryRef;
  input: string;
}

export interface DataTransformArgs {
  expression: string;
}

export interface SetStateArgs {
  values: Record<string, string>;
}

export interface InterruptArgs {
  value?: string;
}

export type NodeType =
  | 'call_llm'
  | 'tool_node'
  | 'call_agent'
  | 'data_transform'
  | 'set_state'
  | 'interrupt';

export type NodeArgs =
  | CallLLMArgs
  | ToolNodeArgs
  | CallAgentArgs
  | DataTransformArgs
  | SetStateArgs
  | InterruptArgs;

export interface NodeDef {
  name: string;
  type: NodeType;
  args: NodeArgs;
}

export interface BranchDef {
  condition: string;
  target: string;
}

export interface EdgeDef {
  source: string;
  target?: string;
  branches?: BranchDef[];
  default?: string;
}

export interface GraphDef {
  entry_point: string;
  nodes: NodeDef[];
  edges: EdgeDef[];
}

export interface LangGraphConfigDef {
  recursion_limit?: number;
  max_concurrency?: number;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface AgentDef {
  state: StateDef;
  graph: GraphDef;
  langgraph_config?: LangGraphConfigDef;
  input_schema?: Record<string, unknown>;
  output_schema?: Record<string, unknown>;
}

export interface LLMDef {
  id: string;
  version?: string;
  provider?: string;
  model_name: string;
}

export interface ToolDef {
  id: string;
  version?: string;
  import_path?: string;
  url?: string;
  protocol?: string;
}

export interface PromptDef {
  id: string;
  version?: string;
  instructions: string;
}

export interface SkillDef {
  id: string;
  version?: string;
  url?: string;
  skill_card_path?: string;
}

export interface HookDef {
  import_path: string;
}

export interface SubAgentDef {
  id: string;
  version?: string;
  url?: string;
  import_path?: string;
  yaml_path?: string;
}

export interface CheckpointerDef {
  type: 'memory';
}

export interface DeclarativeConfig {
  agents: Record<string, AgentDef>;
  llms: LLMDef[];
  tools: ToolDef[];
  prompts: PromptDef[];
  skills: SkillDef[];
  hooks: HookDef[];
  sub_agents: SubAgentDef[];
  checkpointer?: CheckpointerDef;
}

/** Node color mapping */
export const NODE_COLORS: Record<NodeType, string> = {
  call_llm: '#3b82f6',
  tool_node: '#22c55e',
  call_agent: '#a855f7',
  data_transform: '#f97316',
  set_state: '#eab308',
  interrupt: '#ef4444',
};

export const NODE_TYPE_LABELS: Record<NodeType, string> = {
  call_llm: 'Call LLM',
  tool_node: 'Tool Node',
  call_agent: 'Call Agent',
  data_transform: 'Data Transform',
  set_state: 'Set State',
  interrupt: 'Interrupt',
};

export function defaultArgsForType(type: NodeType): NodeArgs {
  switch (type) {
    case 'call_llm':
      return { llm: { id: '', version: '*' }, prompt: '' } as CallLLMArgs;
    case 'tool_node':
      return {} as ToolNodeArgs;
    case 'call_agent':
      return { agent: { id: '', version: '*' }, input: '' } as CallAgentArgs;
    case 'data_transform':
      return { expression: '' } as DataTransformArgs;
    case 'set_state':
      return { values: {} } as SetStateArgs;
    case 'interrupt':
      return {} as InterruptArgs;
  }
}
