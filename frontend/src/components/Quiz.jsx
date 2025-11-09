import React, { useState } from "react";
import { submitAnswers } from "../services/api";
function Quiz({ quiz, onSubmitResult, onBack }) {
  const [answers, setAnswers] = useState({});
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");   

  function handleChange(questionId, value) {
    setAnswers(prev => ({ ...prev, [questionId]: value }));
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const payload = Object.entries(answers).map(([questionId, answer]) => ({
        questionId,
        answer
      }));
      const res = await submitAnswers(quiz.quizId || quiz.quizId, payload);
      onSubmitResult && onSubmitResult(res);
    } catch (e) {
      setError(e.message || "Submit failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="i shld put scroll">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">{quiz.title || "Generated Quiz"}</h2>
        <button onClick={onBack} className="text-sm text-slate-600 underline">Upload another</button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {quiz.questions.map((q, idx) => (
          <div key={q.id} className="p-4 border rounded bg-slate-50">
            <div className="font-medium mb-2">Q{idx + 1}. {q.text}</div>

            {q.type === "mcq" && q.options && (
              <div className="space-y-2">
                {q.options.map((opt, i) => (
                  <label key={i} className="flex items-center gap-2">
                    <input
                      type="radio"
                      name={q.id}
                      value={opt}
                      checked={answers[q.id] === opt}
                      onChange={() => handleChange(q.id, opt)}
                      className="form-radio"
                    />
                    <span>{opt}</span>
                  </label>
                ))}
              </div>
            )}

            {q.type === "open" && (
              <textarea
                rows={4}
                value={answers[q.id] || ""}
                onChange={e => handleChange(q.id, e.target.value)}
                className="w-full mt-2 p-2 border rounded resize-y"
              />
            )}
          </div>
        ))}

        {error && <div className="text-red-600 text-sm">{error}</div>}

        <div className="flex gap-2">
          <button type="submit" disabled={submitting} className="px-4 py-2 bg-indigo-600 text-white rounded">
            {submitting ? "Submitting..." : "Submit Answers"}
          </button>
        </div>
      </form>
    </div>
  );
}
export default Quiz;