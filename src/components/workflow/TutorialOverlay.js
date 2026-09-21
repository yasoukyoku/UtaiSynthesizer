import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import "./TutorialOverlay.css";
export function TutorialOverlay({ onComplete, onSkip } = {}) {
    const { t } = useTranslation();
    const [currentStep, setCurrentStep] = useState(0);
    const [isVisible, setIsVisible] = useState(true);
    const steps = [
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
        }
        else {
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
    if (!step)
        return null;
    const progress = ((currentStep + 1) / steps.length) * 100;
    return (_jsxs("div", { className: "tutorial-overlay", children: [_jsx("div", { className: "tutorial-backdrop", onClick: handleSkip }), _jsxs("div", { className: "tutorial-card", children: [_jsxs("div", { className: "tutorial-header", children: [_jsxs("div", { className: "tutorial-step-indicator", children: [_jsx("span", { className: "tutorial-step-current", children: currentStep + 1 }), _jsx("span", { className: "tutorial-step-separator", children: "/" }), _jsx("span", { className: "tutorial-step-total", children: steps.length })] }), _jsx("button", { className: "tutorial-skip-btn", onClick: handleSkip, children: t("workflow.tutorial.skipButton") })] }), _jsx("div", { className: "tutorial-progress-bar", children: _jsx("div", { className: "tutorial-progress-fill", style: { width: `${progress}%` } }) }), _jsxs("div", { className: "tutorial-content", children: [_jsx("h3", { className: "tutorial-title", children: step.title }), _jsx("p", { className: "tutorial-description", children: step.description }), step.action && (_jsxs("div", { className: "tutorial-action", children: [_jsx("span", { className: "tutorial-action-icon", children: "\uD83D\uDC46" }), _jsx("span", { className: "tutorial-action-text", children: step.action })] }))] }), _jsxs("div", { className: "tutorial-footer", children: [_jsx("button", { className: "tutorial-nav-btn tutorial-prev-btn", onClick: handlePrevious, disabled: currentStep === 0, children: t("workflow.tutorial.previousButton") }), _jsx("div", { className: "tutorial-dots", children: steps.map((_, index) => (_jsx("button", { className: `tutorial-dot ${index === currentStep ? "active" : ""}`, onClick: () => setCurrentStep(index) }, index))) }), _jsx("button", { className: "tutorial-nav-btn tutorial-next-btn", onClick: handleNext, children: currentStep === steps.length - 1
                                    ? t("workflow.tutorial.completeButton")
                                    : t("workflow.tutorial.nextButton") })] })] })] }));
}
export function shouldShowTutorial() {
    return localStorage.getItem("workflow-tutorial-completed") !== "true";
}
export function resetTutorial() {
    localStorage.removeItem("workflow-tutorial-completed");
}
