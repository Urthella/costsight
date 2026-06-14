import { Construction } from "lucide-react";
import { Card, CardBody } from "../components/ui";

/** Stand-in for views not yet migrated from the Streamlit app. */
export default function Placeholder({ name }: { name: string }) {
  return (
    <Card>
      <CardBody className="flex flex-col items-center gap-2 py-16 text-center">
        <Construction className="text-muted-foreground" size={32} />
        <div className="text-lg font-semibold">{name}</div>
        <div className="max-w-md text-sm text-muted-foreground">
          This view is being migrated from the Streamlit app. The data is
          already served by <code>/api/snapshot</code> - the React component is
          next in the queue.
        </div>
      </CardBody>
    </Card>
  );
}
