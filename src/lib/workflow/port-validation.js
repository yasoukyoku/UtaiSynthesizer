import { PORT_COMPATIBILITY } from '@/types/workflow-ports';
export function validateConnection(sourcePort, targetPort) {
    if (targetPort.multiple === false && targetPort.id === sourcePort.id) {
        return {
            valid: false,
            reason: '目标端口不支持多路输入'
        };
    }
    const compatibleTypes = PORT_COMPATIBILITY[sourcePort.type] || [];
    if (!compatibleTypes.includes(targetPort.type)) {
        return {
            valid: false,
            reason: `类型不兼容: ${sourcePort.type} 无法连接到 ${targetPort.type}`
        };
    }
    return { valid: true };
}
export function getPortColor(portType) {
    const colors = {
        audio: '#3b82f6',
        midi: '#10b981',
        chords: '#f59e0b',
        lyrics: '#8b5cf6',
        report: '#6b7280',
        image: '#ec4899',
        video: '#f43f5e',
        any: '#64748b'
    };
    return colors[portType] || colors.any;
}
export function canConnect(sourceType, targetType) {
    const compatibleTypes = PORT_COMPATIBILITY[sourceType] || [];
    return compatibleTypes.includes(targetType);
}
