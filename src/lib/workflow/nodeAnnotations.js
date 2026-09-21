export function createAnnotation(nodeId, content) {
    const now = Date.now();
    return {
        nodeId,
        content,
        color: "yellow",
        createdAt: now,
        updatedAt: now,
    };
}
export function updateAnnotation(annotation, updates) {
    return {
        ...annotation,
        ...updates,
        updatedAt: Date.now(),
    };
}
