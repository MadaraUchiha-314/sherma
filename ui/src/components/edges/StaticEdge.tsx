import { BaseEdge, getSmoothStepPath, type EdgeProps } from '@xyflow/react';

export function StaticEdge(props: EdgeProps) {
  const { sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, markerEnd, style } = props;
  const [edgePath] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  return <BaseEdge path={edgePath} markerEnd={markerEnd} style={{ strokeWidth: 2, stroke: '#6b7280', ...style }} />;
}
