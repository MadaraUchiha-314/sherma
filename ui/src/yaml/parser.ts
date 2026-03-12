import yaml from 'js-yaml';
import type { DeclarativeConfig } from '../types';

export function parseConfig(yamlStr: string): DeclarativeConfig {
  const raw = yaml.load(yamlStr) as Record<string, unknown>;
  if (!raw || typeof raw !== 'object') {
    throw new Error('Invalid YAML: expected an object');
  }

  const config: DeclarativeConfig = {
    agents: (raw.agents as DeclarativeConfig['agents']) || {},
    llms: (raw.llms as DeclarativeConfig['llms']) || [],
    tools: (raw.tools as DeclarativeConfig['tools']) || [],
    prompts: (raw.prompts as DeclarativeConfig['prompts']) || [],
    skills: (raw.skills as DeclarativeConfig['skills']) || [],
    hooks: (raw.hooks as DeclarativeConfig['hooks']) || [],
    sub_agents: (raw.sub_agents as DeclarativeConfig['sub_agents']) || [],
    checkpointer: raw.checkpointer as DeclarativeConfig['checkpointer'],
  };

  // Ensure node args are objects (handle `args: {}` edge case)
  for (const agent of Object.values(config.agents)) {
    for (const node of agent.graph.nodes) {
      if (!node.args) {
        node.args = {} as never;
      }
    }
  }

  return config;
}
