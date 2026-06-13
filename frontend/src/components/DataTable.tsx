import type { ReactNode } from "react";

export interface Column {
  key: string;
  label: string;
  align?: "left" | "right";
  render?: (row: Record<string, unknown>) => ReactNode;
}

export function DataTable({
  columns,
  rows,
  maxRows,
  empty = "No rows.",
}: {
  columns: Column[];
  rows: Record<string, unknown>[];
  maxRows?: number;
  empty?: string;
}) {
  const shown = maxRows ? rows.slice(0, maxRows) : rows;
  if (!rows.length)
    return <div className="py-8 text-center text-sm text-muted-foreground">{empty}</div>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left text-muted-foreground">
            {columns.map((c) => (
              <th
                key={c.key}
                className={
                  "py-1.5 pr-4 font-medium " +
                  (c.align === "right" ? "text-right" : "")
                }
              >
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {shown.map((row, i) => (
            <tr key={i} className="border-b border-border/50 hover:bg-muted/40">
              {columns.map((c) => (
                <td
                  key={c.key}
                  className={
                    "py-1.5 pr-4 " + (c.align === "right" ? "text-right tabular-nums" : "")
                  }
                >
                  {c.render ? c.render(row) : String(row[c.key] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {maxRows && rows.length > maxRows && (
        <div className="pt-2 text-xs text-muted-foreground">
          Showing {maxRows} of {rows.length} rows.
        </div>
      )}
    </div>
  );
}
