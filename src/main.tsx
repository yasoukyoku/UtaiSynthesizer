import React from "react";
// 首帧主题注入 — 避免 FOUC (Flash of Unstyled Content)
(function(){ try {
  var t = localStorage.getItem("utai.theme") || "dark";
  document.documentElement.setAttribute("data-theme", t);
} catch(e){} })();

import ReactDOM from "react-dom/client";
import { App } from "./App";
import "./styles/global.css";
import "./i18n";
import { loadSetting } from "./lib/settings";

document.addEventListener("contextmenu", (e) => e.preventDefault());

// Apply the persisted built-in skin before first paint so it never flashes a wrong theme.
// New installs default to the violet "junzi" skin (the brand default); an explicit choice
// saved by the user always wins.
document.documentElement.dataset.skin = loadSetting<string>("utai.skin", "junzi");

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
