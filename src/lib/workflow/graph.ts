import type { Workflow, WorkflowNode, WorkflowConnection } from "../../types/project";

export interface GraphNode {
  node: WorkflowNode;
  inEdges: WorkflowConnection[];
  outEdges: WorkflowConnection[];
}

export interface ParsedGraph {
  nodes: Map<string, GraphNode>;
  sorted: string[];
  inputNodeId: string;
  outputNodeIds: string[];
}

export function parseWorkflowGraph(workflow: Workflow): ParsedGraph {
  const nodes = new Map<string, GraphNode>();

  for (const node of workflow.nodes) {
    nodes.set(node.id, { node, inEdges: [], outEdges: [] });
  }

  for (const conn of workflow.connections) {
    nodes.get(conn.fromNode)?.outEdges.push(conn);
    nodes.get(conn.toNode)?.inEdges.push(conn);
  }

  const inputNodeId = workflow.nodes.find((n) => n.nodeType === "input")?.id;
  if (!inputNodeId) throw new Error("Workflow has no input node");

  const outputNodeIds = workflow.nodes
    .filter((n) => n.nodeType === "output")
    .map((n) => n.id);
  if (outputNodeIds.length === 0) throw new Error("Workflow has no output nodes");

  const sorted = topologicalSort(nodes);

  return { nodes, sorted, inputNodeId, outputNodeIds };
}

function topologicalSort(nodes: Map<string, GraphNode>): string[] {
  const inDegree = new Map<string, number>();
  for (const [id, gn] of nodes) {
    inDegree.set(id, gn.inEdges.length);
  }

  const queue: string[] = [];
  for (const [id, deg] of inDegree) {
    if (deg === 0) queue.push(id);
  }

  const result: string[] = [];
  while (queue.length > 0) {
    const id = queue.shift()!;
    result.push(id);
    const gn = nodes.get(id)!;
    for (const edge of gn.outEdges) {
      const newDeg = (inDegree.get(edge.toNode) ?? 1) - 1;
      inDegree.set(edge.toNode, newDeg);
      if (newDeg === 0) queue.push(edge.toNode);
    }
  }

  if (result.length !== nodes.size) {
    throw new Error("Workflow contains a cycle");
  }

  return result;
}
