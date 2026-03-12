import { useState } from 'react';

interface ResourceListProps<T> {
  title: string;
  items: T[];
  getId: (item: T) => string;
  getLabel: (item: T) => string;
  onAdd: (item: T) => void;
  onUpdate: (index: number, item: T) => void;
  onDelete: (index: number) => void;
  renderForm: (item: T | null, onSave: (item: T) => void, onCancel: () => void) => React.ReactNode;
}

export function ResourceList<T>({
  title,
  items,
  getId,
  getLabel,
  onAdd,
  onUpdate,
  onDelete,
  renderForm,
}: ResourceListProps<T>) {
  const [editing, setEditing] = useState<{ index: number; item: T } | null>(null);
  const [adding, setAdding] = useState(false);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">{title}</h3>
        <button
          onClick={() => setAdding(true)}
          className="text-xs text-blue-600 hover:text-blue-800"
        >
          + Add
        </button>
      </div>

      {adding &&
        renderForm(
          null,
          (item) => {
            onAdd(item);
            setAdding(false);
          },
          () => setAdding(false)
        )}

      {items.map((item, idx) => (
        <div key={getId(item)} className="flex items-center justify-between px-2 py-1.5 rounded bg-white border border-gray-200 text-sm">
          {editing?.index === idx ? (
            <div className="w-full">
              {renderForm(
                editing.item,
                (updated) => {
                  onUpdate(idx, updated);
                  setEditing(null);
                },
                () => setEditing(null)
              )}
            </div>
          ) : (
            <>
              <span className="text-gray-700 truncate">{getLabel(item)}</span>
              <div className="flex gap-1 shrink-0">
                <button
                  onClick={() => setEditing({ index: idx, item })}
                  className="text-xs text-gray-500 hover:text-blue-600"
                >
                  Edit
                </button>
                <button
                  onClick={() => onDelete(idx)}
                  className="text-xs text-gray-500 hover:text-red-600"
                >
                  Del
                </button>
              </div>
            </>
          )}
        </div>
      ))}

      {items.length === 0 && !adding && (
        <p className="text-xs text-gray-400 italic">No {title.toLowerCase()} defined</p>
      )}
    </div>
  );
}
