import React from "react";

export default function Result({ result, onRestart }) {
  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Results</h2>
      <div className="p-4 border rounded bg-slate-50 mb-4">
        <div className="text-lg">Score: {result.score} / {result.total}</div>
        {result.feedback && <div className="mt-2 text-slate-700">{result.feedback}</div>}
      </div>

      <div className="flex gap-2">
        <button onClick={onRestart} className="px-4 py-2 bg-slate-700 text-white rounded">Try another PDF</button>
      </div>
    </div>
  );
}
