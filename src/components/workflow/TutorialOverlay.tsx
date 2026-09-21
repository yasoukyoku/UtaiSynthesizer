import { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import "./TutorialOverlay.css";

interface TutorialStep {
  title: string;
  description: string;
  target?: string;
  highlightClass?: string;
  action?: string;
}

interface Props {
  onComplete?: () => void;
  onSkip?: () => void;
}

export function TutorialOverlay({ onComplete, onSkip }: Props = {}) {
  const { t } = useTranslation();
  const [currentStep, setCurrentStep] = useState(0);
  const [isVisible, setIsVisible] = useState(true);

  const steps: TutorialStep[] = [
    {
      title: t("workflow.tutorial.step1.title"),
      description: t("workflow.tutorial.step1.description"),
      target: ".node-palette",
      highlightClass: "tutorial-highlight-palette",
    },
    {
      title: t("workflow.tutorial.step2.title"),
      description: t("workflow.tutorial.step2.description"),
      action: t("workflow.tutorial.step2.action"),
    },
    {
      title: t("workflow.tutorial.step3.title"),
      description: t("workflow.tutorial.step3.description"),
      target: ".react-flow__node",
      highlightClass: "tutorial-highlight-node",
    },
    {
      title: t("workflow.tutorial.step4.title"),
      description: t("workflow.tutorial.step4.description"),
    },
    {
      title: t("workflow.tutorial.step5.title"),
      description: t("workflow.tutorial.step5.description"),
      target: ".react-flow__controls",
      highlightClass: "tutorial-highlight-controls",
    },
  ];

  useEffect(() => {
    const step = steps[currentStep];
    if (step?.highlightClass) {
      document.body.classList.add(step.highlightClass);
    }

    return () => {
      steps.forEach((step) => {
        if (step?.highlightClass) {
          document.body.classList.remove(step.highlightClass);
        }
      });
    };
  }, [currentStep]);

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      handleComplete();
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleComplete = () => {
    setIsVisible(false);
    localStorage.setItem("workflow-tutorial-completed", "true");
    onComplete?.();
  };

  const handleSkip = () => {
    setIsVisible(false);
    localStorage.setItem("workflow-tutorial-completed", "true");
    onSkip?.();
  };

  if (!isVisible) {
    return null;
  }

  const step = steps[currentStep];
  if (!step) return null;
  const progress = ((currentStep + 1) / steps.length) * 100;

  return (
    <div className="tutorial-overlay">
      <div className="tutorial-backdrop" onClick={handleSkip} />
      
      <div className="tutorial-card">
        <div className="tutorial-header">
          <div className="tutorial-step-indicator">
            <span className="tutorial-step-current">{currentStep + 1}</span>
            <span className="tutorial-step-separator">/</span>
            <span className="tutorial-step-total">{steps.length}</span>
          </div>
          <button className="tutorial-skip-btn" onClick={handleSkip}>
            {t("workflow.tutorial.skipButton")}
          </button>
        </div>

        <div className="tutorial-progress-bar">
          <div className="tutorial-progress-fill" style={{ width: `${progress}%` }} />
        </div>

        <div className="tutorial-content">
          <h3 className="tutorial-title">{step.title}</h3>
          <p className="tutorial-description">{step.description}</p>
          {step.action && (
            <div className="tutorial-action">
              <span className="tutorial-action-icon">👆</span>
              <span className="tutorial-action-text">{step.action}</span>
            </div>
          )}
        </div>

        <div className="tutorial-footer">
          <button 
            className="tutorial-nav-btn tutorial-prev-btn" 
            onClick={handlePrevious}
            disabled={currentStep === 0}
          >
            {t("workflow.tutorial.previousButton")}
          </button>
          <div className="tutorial-dots">
            {steps.map((_, index) => (
              <button
                key={index}
                className={`tutorial-dot ${index === currentStep ? "active" : ""}`}
                onClick={() => setCurrentStep(index)}
              />
            ))}
          </div>
          <button 
            className="tutorial-nav-btn tutorial-next-btn" 
            onClick={handleNext}
          >
            {currentStep === steps.length - 1 
              ? t("workflow.tutorial.completeButton") 
              : t("workflow.tutorial.nextButton")}
          </button>
        </div>
      </div>
    </div>
  );
}

export function shouldShowTutorial(): boolean {
  return localStorage.getItem("workflow-tutorial-completed") !== "true";
}

export function resetTutorial(): void {
  localStorage.removeItem("workflow-tutorial-completed");
}
