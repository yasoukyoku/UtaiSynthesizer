import { useState } from "react";
import { useTranslation } from "react-i18next";
import { loadSetting, saveSetting } from "../../lib/settings";
import "./OnboardingTour.css";

/**
 * 首次启动引导（新手友好）：4 步卡片向导，讲清「导入 → 选歌手 → 渲染 → AI 编曲」主流程。
 *
 * · 只在从未看过时出现一次（localStorage `utai.onboardingDone` 标记）。
 * · 纯居中卡片 + 步骤圆点，不遮罩不高亮——保持极简，随时可跳过。
 * · 帮助菜单的「使用说明」承载完整文档，这里只做主流程的第一次见面。
 */
const STORAGE_KEY = "utai.onboardingDone";

/** Step content lives in i18n (`onboarding.stepN.title/body`); the tour is pure chrome. */
const STEP_KEYS = ["step1", "step2", "step3", "step4", "step5"] as const;

export function OnboardingTour() {
  const { t } = useTranslation();
  const [step, setStep] = useState(0);
  const [open] = useState(() => !loadSetting(STORAGE_KEY, false));
  const [dismissed, setDismissed] = useState(false);
  if (!open || dismissed) return null;

  const close = () => {
    saveSetting(STORAGE_KEY, true);
    setDismissed(true);
  };
  const last = step === STEP_KEYS.length - 1;
  const key = STEP_KEYS[step]!;

  return (
    <div className="onboard-overlay" role="dialog" aria-modal="true" aria-label={t("onboarding.title")}>
      <div className="onboard-card">
        <div className="onboard-logo" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="34" height="34">
            <defs>
              <linearGradient id="onboardGrad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stopColor="var(--brand-grad-a, #c4b5fd)" />
                <stop offset="1" stopColor="var(--brand-grad-b, #8b5cf6)" />
              </linearGradient>
            </defs>
            <rect x="1" y="1" width="22" height="22" rx="6.5" fill="url(#onboardGrad)" />
            <g fill="#fff">
              <rect x="6.6" y="14" width="3" height="3.4" rx="1.5" />
              <rect x="13.8" y="13.2" width="3" height="3.4" rx="1.5" />
              <rect x="8.6" y="5.5" width="1.2" height="8.8" />
              <rect x="15.8" y="4.8" width="1.2" height="8.8" />
              <path d="M8.4 5.4h7.4v1.3H8.4z" />
            </g>
          </svg>
        </div>
        <h2 className="onboard-title">{t("onboarding.title")}</h2>

        <div className="onboard-step">
          <div className="onboard-step-head">
            <span className="onboard-step-num">{step + 1}</span>
            <h3>{t(`onboarding.${key}.title`)}</h3>
          </div>
          <p>{t(`onboarding.${key}.body`)}</p>
        </div>

        <div className="onboard-dots" aria-hidden="true">
          {STEP_KEYS.map((k, i) => (
            <span key={k} className={`onboard-dot ${i === step ? "active" : ""}`} />
          ))}
        </div>

        <div className="onboard-actions">
          <button className="onboard-btn ghost" onClick={close}>
            {t("onboarding.skip")}
          </button>
          <div className="onboard-actions-right">
            {step > 0 && (
              <button className="onboard-btn ghost" onClick={() => setStep((s) => s - 1)}>
                {t("onboarding.prev")}
              </button>
            )}
            <button
              className="onboard-btn primary"
              onClick={() => (last ? close() : setStep((s) => s + 1))}
            >
              {last ? t("onboarding.start") : t("onboarding.next")}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
