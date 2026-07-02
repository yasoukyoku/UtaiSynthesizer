import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import zh from "./zh.json";
import en from "./en.json";
import ja from "./ja.json";

const savedLang =
  typeof localStorage !== "undefined" ? localStorage.getItem("lang") : null;

i18n.use(initReactI18next).init({
  resources: {
    zh: { translation: zh },
    en: { translation: en },
    ja: { translation: ja },
  },
  lng: savedLang ?? "zh",
  fallbackLng: "en",
  interpolation: { escapeValue: false },
});

export default i18n;
