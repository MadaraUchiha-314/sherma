import type { EdgeTypes } from '@xyflow/react';
import { StaticEdge } from './StaticEdge';
import { ConditionalEdge } from './ConditionalEdge';

export const edgeTypes: EdgeTypes = {
  staticEdge: StaticEdge,
  conditionalEdge: ConditionalEdge,
};
