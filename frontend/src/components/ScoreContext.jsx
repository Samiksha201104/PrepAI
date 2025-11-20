import { createContext, useContext, useState,useRef } from "react";

const ScoreContext = createContext();

export function ScoreProvider({ children }) {
  const [score, setScore] = useState(0);
  const clickSound = useRef(new Audio("/sounds/click.mp3")).current;


  return (
    <ScoreContext.Provider value={{ score, setScore,clickSound }}>
      {children}
    </ScoreContext.Provider>
  );
}

export function useScore() {
  return useContext(ScoreContext);
}
