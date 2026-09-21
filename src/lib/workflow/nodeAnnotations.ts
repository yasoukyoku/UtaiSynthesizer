export interface NodeAnnotation {
  nodeId: string;
  content: string;
  color?: "yellow" | "blue" | "green" | "red" | "purple";
  createdAt: number;
  updatedAt: number;
}

export function createAnnotation(nodeId: string, content: string): NodeAnnotation {
  const now = Date.now();
  return {
    nodeId,
    content,
    color: "yellow",
    createdAt: now,
    updatedAt: now,
  };
}

export function updateAnnotation(annotation: NodeAnnotation, updates: Partial<Omit<NodeAnnotation, "nodeId" | "createdAt">>): NodeAnnotation {
  return {
    ...annotation,
    ...updates,
    updatedAt: Date.now(),
  };
}
