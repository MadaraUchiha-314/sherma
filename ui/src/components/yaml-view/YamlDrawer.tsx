import { useStore } from '../../store';
import { serializeConfig } from '../../yaml/serializer';

export function YamlDrawer() {
  const yamlDrawerOpen = useStore((s) => s.yamlDrawerOpen);
  const toggleYamlDrawer = useStore((s) => s.toggleYamlDrawer);
  const config = useStore((s) => s.config);

  if (!yamlDrawerOpen) return null;

  const yamlStr = serializeConfig(config);

  return (
    <div className="fixed inset-0 z-50 flex">
      <div className="absolute inset-0 bg-black/30" onClick={toggleYamlDrawer} />
      <div className="relative ml-auto w-[600px] bg-white shadow-xl flex flex-col">
        <div className="flex items-center justify-between p-3 border-b border-gray-200">
          <span className="font-semibold text-sm">Full Config YAML</span>
          <button onClick={toggleYamlDrawer} className="text-gray-500 hover:text-gray-700 text-lg">
            &times;
          </button>
        </div>
        <pre className="flex-1 overflow-auto p-4 text-xs font-mono text-gray-800 bg-gray-50">
          {yamlStr}
        </pre>
      </div>
    </div>
  );
}
