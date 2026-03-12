import type { NodeProps } from '@xyflow/react';
import { BaseNode } from './BaseNode';
import type { NodeDef } from '../../types';

export function CustomNode({ data, selected }: NodeProps) {
  const nodeDef = data.nodeDef as NodeDef;
  return <BaseNode nodeDef={nodeDef} selected={selected} />;
}
