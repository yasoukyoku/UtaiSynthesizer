import { jsxs as _jsxs, jsx as _jsx } from "react/jsx-runtime";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { exampleWorkflows } from "../../lib/workflow/exampleWorkflows";
import "./ExampleWorkflowLibrary.css";
export function ExampleWorkflowLibrary({ onLoad, onClose }) {
    const { t, i18n } = useTranslation();
    const [selectedCategory, setSelectedCategory] = useState("all");
    const [selectedDifficulty, setSelectedDifficulty] = useState("all");
    const filteredWorkflows = exampleWorkflows.filter((ex) => {
        if (selectedCategory !== "all" && ex.category !== selectedCategory)
            return false;
        if (selectedDifficulty !== "all" && ex.difficulty !== selectedDifficulty)
            return false;
        return true;
    });
    const categoryLabels = {
        all: t("workflow.exampleLibrary.allCategories"),
        basic: t("workflow.exampleLibrary.categoryBasic"),
        advanced: t("workflow.exampleLibrary.categoryAdvanced"),
        effect: t("workflow.exampleLibrary.categoryEffect"),
        analysis: t("workflow.exampleLibrary.categoryAnalysis"),
    };
    const difficultyLabels = {
        all: t("workflow.exampleLibrary.allDifficulties"),
        beginner: t("workflow.exampleLibrary.difficultyBeginner"),
        intermediate: t("workflow.exampleLibrary.difficultyIntermediate"),
        advanced: t("workflow.exampleLibrary.difficultyAdvanced"),
    };
    const difficultyIcons = {
        beginner: "🟢",
        intermediate: "🟡",
        advanced: "🔴",
    };
    return (_jsx("div", { className: "example-library-overlay", onClick: onClose, children: _jsxs("div", { className: "example-library-panel", onClick: (e) => e.stopPropagation(), children: [_jsxs("div", { className: "example-library-header", children: [_jsxs("h2", { children: ["\uD83D\uDCDA ", t("workflow.exampleLibrary.title")] }), _jsx("button", { className: "example-library-close", onClick: onClose, children: "\u2715" })] }), _jsxs("div", { className: "example-library-filters", children: [_jsxs("div", { className: "example-library-filter-group", children: [_jsxs("label", { children: [t("workflow.exampleLibrary.filterCategory"), ":"] }), _jsx("select", { value: selectedCategory, onChange: (e) => setSelectedCategory(e.target.value), children: Object.entries(categoryLabels).map(([key, label]) => (_jsx("option", { value: key, children: label }, key))) })] }), _jsxs("div", { className: "example-library-filter-group", children: [_jsxs("label", { children: [t("workflow.exampleLibrary.filterDifficulty"), ":"] }), _jsx("select", { value: selectedDifficulty, onChange: (e) => setSelectedDifficulty(e.target.value), children: Object.entries(difficultyLabels).map(([key, label]) => (_jsx("option", { value: key, children: label }, key))) })] })] }), _jsx("div", { className: "example-library-grid", children: filteredWorkflows.map((example) => (_jsxs("div", { className: "example-library-card", children: [_jsxs("div", { className: "example-library-card-header", children: [_jsx("h3", { children: i18n.language === "en" ? example.nameEn : example.name }), _jsx("span", { className: "example-library-difficulty", title: difficultyLabels[example.difficulty], children: difficultyIcons[example.difficulty] })] }), _jsx("p", { className: "example-library-description", children: i18n.language === "en" ? example.descriptionEn : example.description }), _jsxs("div", { className: "example-library-card-footer", children: [_jsxs("span", { className: "example-library-node-count", children: ["\uD83D\uDD17 ", example.workflow.nodes.length, " ", t("workflow.exampleLibrary.nodes")] }), _jsx("button", { className: "example-library-load-btn", onClick: () => {
                                            if (onLoad)
                                                onLoad(example);
                                            onClose();
                                        }, children: t("workflow.exampleLibrary.loadButton") })] })] }, example.id))) }), filteredWorkflows.length === 0 && (_jsx("div", { className: "example-library-empty", children: _jsx("p", { children: t("workflow.exampleLibrary.noResults") }) }))] }) }));
}
