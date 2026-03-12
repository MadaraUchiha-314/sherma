import type { NodeTypes } from '@xyflow/react';
import { CustomNode } from './CustomNode';
import { StartNode } from './StartNode';
import { EndNode } from './EndNode';

export const nodeTypes: NodeTypes = {
  customNode: CustomNode,
  startNode: StartNode,
  endNode: EndNode,
};
