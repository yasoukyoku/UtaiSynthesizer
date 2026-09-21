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

  // Both endpoints must exist or the edge is DROPPED. Registering the two halves independently (the old
  // `nodes.get(a)?.outEdges…; nodes.get(b)?.inEdges…`) let a connection whose fromNode was deleted still
  // raise toNode's in-degree with no matching out-edge to ever decrement it — the node and every successor
  // stayed unreachable and topologicalSort blamed "Workflow contains a cycle" for a graph with no visible
  // cycle at all. A half-attached edge carries no data either way, so skipping it is the honest reading.
  for (const conn of workflow.connections) {
    const from = nodes.get(conn.fromNode);
    const to = nodes.get(conn.toNode);
    if (!from || !to) continue;
    from.outEdges.push(conn);
    to.inEdges.push(conn);
  }

  const inputNodes = workflow.nodes.filter((n) => n.nodeType === "input");
  const inputNodeId = inputNodes[0]?.id;
  if (!inputNodeId) throw new Error("Workflow has no input node");
  // Only inputNodeId is ever seeded with the segment audio, so a 2nd input node feeds its whole subtree
  // nothing and every consumer failed with the misleading `has no input connected`. Name the real cause.
  if (inputNodes.length > 1) throw new Error("Workflow has more than one input node");

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
