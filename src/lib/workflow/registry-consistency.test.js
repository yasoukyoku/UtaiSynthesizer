import { describe, it, expect } from "vitest";
import { NODE_PORTS } from "./ports";
import { NODE_DAMAGE } from "./damage";
import { rfTypeToWfType, wfTypeToRfType } from "./rfTypes";
describe("四张注册表一致性回归测试（防止 Bug #1 lufsAnalyze 类似漏洞再现）", () => {
    const allWfTypes = Object.keys(NODE_PORTS);
    const rfKeys = Object.keys(rfTypeToWfType);
    const wfKeys = Object.keys(wfTypeToRfType);
    it("NODE_PORTS 与 NODE_DAMAGE 的 key 集合必须完全一致", () => {
        const portsKeys = new Set(Object.keys(NODE_PORTS));
        const damageKeys = new Set(Object.keys(NODE_DAMAGE));
        const onlyInPorts = [...portsKeys].filter(k => !damageKeys.has(k));
        const onlyInDamage = [...damageKeys].filter(k => !portsKeys.has(k));
        expect(onlyInPorts, `NODE_PORTS 有但 NODE_DAMAGE 缺失: ${onlyInPorts.join(", ")}`).toEqual([]);
        expect(onlyInDamage, `NODE_DAMAGE 有但 NODE_PORTS 缺失: ${onlyInDamage.join(", ")}`).toEqual([]);
        expect(portsKeys.size).toBe(damageKeys.size);
    });
    it("wfTypeToRfType 必须覆盖所有 WorkflowNodeType（除历史别名 msst）", () => {
        const missing = allWfTypes.filter(wf => !wfKeys.includes(wf));
        expect(missing, `wfTypeToRfType 缺失映射: ${missing.join(", ")}`).toEqual([]);
    });
    it("rfTypeToWfType 的所有值必须是合法的 WorkflowNodeType", () => {
        const validWfTypes = new Set(allWfTypes);
        const invalidMappings = rfKeys
            .map(rf => ({ rf, wf: rfTypeToWfType[rf] }))
            .filter(({ wf }) => !validWfTypes.has(wf));
        expect(invalidMappings, `rfTypeToWfType 映射到不存在的 WorkflowNodeType: ${invalidMappings.map(m => `${m.rf} -> ${m.wf}`).join(", ")}`).toEqual([]);
    });
    it("双向映射必须可逆（wf -> rf -> wf 往返不变）", () => {
        const broken = wfKeys
            .map(wf => {
            const rf = wfTypeToRfType[wf];
            const back = rf ? rfTypeToWfType[rf] : undefined;
            return { wf, rf, back };
        })
            .filter(({ wf, back }) => back !== wf);
        expect(broken, `双向映射破损: ${broken.map(b => `${b.wf} -> ${b.rf} -> ${b.back}`).join(", ")}`).toEqual([]);
    });
    it("特定分析节点必须同时出现在四张表（防止 lufsAnalyze 类似遗漏）", () => {
        const criticalNodes = [
            "lufsAnalyze",
            "spectrogram",
            "f0Curve",
            "timbreMetrics",
            "harmonicityCheck",
            "spectralCompare",
            "dtwAlign",
            "abCompare",
        ];
        for (const node of criticalNodes) {
            expect(NODE_PORTS[node], `${node} 在 NODE_PORTS 中缺失`).toBeDefined();
            expect(NODE_DAMAGE[node], `${node} 在 NODE_DAMAGE 中缺失`).toBeDefined();
            expect(wfTypeToRfType[node], `${node} 在 wfTypeToRfType 中缺失`).toBeDefined();
            const rf = wfTypeToRfType[node];
            if (rf) {
                expect(rfTypeToWfType[rf], `${node} 的 rf 类型 ${rf} 在 rfTypeToWfType 中缺失`).toBe(node);
            }
        }
    });
    it("NODE_PORTS 声明的节点都应该能在 wfTypeToRfType 找到（除 input/output）", () => {
        const exceptions = new Set(["input", "output"]);
        const unmapped = allWfTypes.filter(wf => !exceptions.has(wf) && !wfKeys.includes(wf));
        expect(unmapped, `这些节点有端口和损伤表但缺 rfTypes 映射: ${unmapped.join(", ")}`).toEqual([]);
    });
});
