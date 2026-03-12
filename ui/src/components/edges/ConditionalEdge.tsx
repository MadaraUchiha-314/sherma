import { BaseEdge, EdgeLabelRenderer, getSmoothStepPath, type EdgeProps } from '@xyflow/react';

export function ConditionalEdge(props: EdgeProps) {
  const { sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, markerEnd, data, style } = props;
  const [edgePath, labelX, labelY] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  const isDefault = data?.isDefault;
  const condition = data?.condition as string | undefined;
  const label = isDefault ? 'default' : condition;

  return (
    <>
      <BaseEdge
        path={edgePath}
        markerEnd={markerEnd}
        style={{ strokeWidth: 2, stroke: isDefault ? '#f59e0b' : '#8b5cf6', strokeDasharray: '5 3', ...style }}
      />
      {label && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: 'all',
            }}
            className="bg-white border border-gray-300 rounded px-2 py-0.5 text-xs text-gray-700 shadow-sm max-w-[200px] truncate"
          >
            {label}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}
