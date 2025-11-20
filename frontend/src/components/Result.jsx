import React from "react";
import { useScore } from "./ScoreContext";

function Result({ questions, summary, onRestart }) {
  const { score } = useScore();
  const { clickSound } = useScore();
  return (
    <div className="space-y-6 max-w-2xl mx-auto">
      <div className="bg-[#E5DDD0] border-3 border-[#99968D] p-8 rounded-4xl shadow-lg text-center">
        <div className="text-6xl mb-4 ">🎉</div>
        <h2 className="text-3xl font-bold text-[#040404] mb-2">
          Quiz Complete!
        </h2>
        <p className="text-gray-600 mb-6">
          Great job! You've finished all the questions.
        </p>
        <p className="text-3xl font-bold text-green-900">{`Your Result : ${score}/${questions.length}`}</p>
      </div>
      <div className="text-center">
        <button
          onClick={() => {
            clickSound.play();
            onRestart();
          }}
          className="bg-[#99968D] text-[#040404] px-8 mb-5 py-3 rounded-full hover:bg-[#040404] hover:text-[#FFFFFF] font-semibold transition-colors shadow-md"
        >
          Upload Another Document
        </button>
      </div>
    </div>
  );
}

export default Result;