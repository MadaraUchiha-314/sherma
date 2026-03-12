import yaml from 'js-yaml';
import type { DeclarativeConfig } from '../types';

/** Remove keys with null/undefined values recursively */
function stripNulls(obj: unknown): unknown {
  if (Array.isArray(obj)) {
    return obj.map(stripNulls);
  }
  if (obj !== null && typeof obj === 'object') {
    const result: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(obj as Record<string, unknown>)) {
      if (value !== null && value !== undefined) {
        const stripped = stripNulls(value);
        // Also skip empty arrays and empty objects for cleaner output
        if (Array.isArray(stripped) && stripped.length === 0) continue;
        if (typeof stripped === 'object' && !Array.isArray(stripped) && Object.keys(stripped as object).length === 0) continue;
        result[key] = stripped;
      }
    }
    return result;
  }
  return obj;
}

export function serializeConfig(config: DeclarativeConfig): string {
  const cleaned = stripNulls(config);
  return yaml.dump(cleaned, {
    lineWidth: 120,
    noRefs: true,
    sortKeys: false,
  });
}

export function serializeNodeYaml(nodeDef: { name: string; type: string; args: unknown }): string {
  const cleaned = stripNulls({ name: nodeDef.name, type: nodeDef.type, args: nodeDef.args });
  return yaml.dump(cleaned, { lineWidth: 120, noRefs: true });
}
