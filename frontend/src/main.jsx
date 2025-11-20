import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";
import { ScoreProvider } from "./components/ScoreContext";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ScoreProvider>    
      <App />
    </ScoreProvider>

  </React.StrictMode>
);
