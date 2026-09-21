import { useState } from "react";
import { useTranslation } from "react-i18next";
import { exampleWorkflows, type ExampleWorkflow } from "../../lib/workflow/exampleWorkflows";
import "./ExampleWorkflowLibrary.css";

interface Props {
  onLoad?: (workflow: ExampleWorkflow) => void;
  onClose: () => void;
}

export function ExampleWorkflowLibrary({ onLoad, onClose }: Props) {
  const { t, i18n } = useTranslation();
  const [selectedCategory, setSelectedCategory] = useState<ExampleWorkflow["category"] | "all">("all");
  const [selectedDifficulty, setSelectedDifficulty] = useState<ExampleWorkflow["difficulty"] | "all">("all");

  const filteredWorkflows = exampleWorkflows.filter((ex) => {
    if (selectedCategory !== "all" && ex.category !== selectedCategory) return false;
    if (selectedDifficulty !== "all" && ex.difficulty !== selectedDifficulty) return false;
    return true;
  });

  const categoryLabels: Record<ExampleWorkflow["category"] | "all", string> = {
    all: t("workflow.exampleLibrary.allCategories"),
    basic: t("workflow.exampleLibrary.categoryBasic"),
    advanced: t("workflow.exampleLibrary.categoryAdvanced"),
    effect: t("workflow.exampleLibrary.categoryEffect"),
    analysis: t("workflow.exampleLibrary.categoryAnalysis"),
  };

  const difficultyLabels: Record<ExampleWorkflow["difficulty"] | "all", string> = {
    all: t("workflow.exampleLibrary.allDifficulties"),
    beginner: t("workflow.exampleLibrary.difficultyBeginner"),
    intermediate: t("workflow.exampleLibrary.difficultyIntermediate"),
    advanced: t("workflow.exampleLibrary.difficultyAdvanced"),
  };

  const difficultyIcons: Record<ExampleWorkflow["difficulty"], string> = {
    beginner: "🟢",
    intermediate: "🟡",
    advanced: "🔴",
  };

  return (
    <div className="example-library-overlay" onClick={onClose}>
      <div className="example-library-panel" onClick={(e) => e.stopPropagation()}>
        <div className="example-library-header">
          <h2>📚 {t("workflow.exampleLibrary.title")}</h2>
          <button className="example-library-close" onClick={onClose}>✕</button>
        </div>

        <div className="example-library-filters">
          <div className="example-library-filter-group">
            <label>{t("workflow.exampleLibrary.filterCategory")}:</label>
            <select 
              value={selectedCategory} 
              onChange={(e) => setSelectedCategory(e.target.value as ExampleWorkflow["category"] | "all")}
            >
              {Object.entries(categoryLabels).map(([key, label]) => (
                <option key={key} value={key}>{label}</option>
              ))}
            </select>
          </div>

          <div className="example-library-filter-group">
            <label>{t("workflow.exampleLibrary.filterDifficulty")}:</label>
            <select 
              value={selectedDifficulty} 
              onChange={(e) => setSelectedDifficulty(e.target.value as ExampleWorkflow["difficulty"] | "all")}
            >
              {Object.entries(difficultyLabels).map(([key, label]) => (
                <option key={key} value={key}>{label}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="example-library-grid">
          {filteredWorkflows.map((example) => (
            <div key={example.id} className="example-library-card">
              <div className="example-library-card-header">
                <h3>{i18n.language === "en" ? example.nameEn : example.name}</h3>
                <span className="example-library-difficulty" title={difficultyLabels[example.difficulty]}>
                  {difficultyIcons[example.difficulty]}
                </span>
              </div>
              <p className="example-library-description">
                {i18n.language === "en" ? example.descriptionEn : example.description}
              </p>
              <div className="example-library-card-footer">
                <span className="example-library-node-count">
                  🔗 {example.workflow.nodes.length} {t("workflow.exampleLibrary.nodes")}
                </span>
                <button 
                  className="example-library-load-btn"
                  onClick={() => {
                    if (onLoad) onLoad(example);
                    onClose();
                  }}
                >
                  {t("workflow.exampleLibrary.loadButton")}
                </button>
              </div>
            </div>
          ))}
        </div>

        {filteredWorkflows.length === 0 && (
          <div className="example-library-empty">
            <p>{t("workflow.exampleLibrary.noResults")}</p>
          </div>
        )}
      </div>
    </div>
  );
}
