import React, { useState } from "react";
import UploadForm from "./components/UploadForm";
import Quiz from "./components/Quiz";
import Result from "./components/Result";

export default function App() {
  const [quizData, setQuizData] = useState(null);
  const [error, setError] = useState("");
  const [showResult, setShowResult] = useState(false);

  const handleRestart = () => {
    setQuizData(null);
    setError("");
    setShowResult(false);
  };

  return (
    <div style={{ backgroundImage: "url('src/assets/bg.jpg')" }} className="min-h-screen flex flex-col items-center p-6 bg-cover bg-center justify-center">
      <div className="max-w-4xl w-full">
        <h1 className="text-5xl font-bold mb-2 text-center text-slate-100">
          PrepAI Quiz Generator
        </h1>
        <p className="text-center text-white mb-8">
          Upload a document and generate AI-powered quizzes and short answers
        </p>

        {!quizData && !showResult && (
          <UploadForm onQuizReady={setQuizData} setError={setError} />
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mt-4">
            {error}
          </div>
        )}

        {quizData && !showResult && (
          <div className="mt-6">
            {/* {quizData.summary && (
              <div className="bg-white p-6 rounded-lg shadow-md mb-6">
                <h2 className="text-2xl font-semibold mb-3 text-indigo-800">
                  📝 Document Summary
                </h2>
                <p className="text-gray-700 whitespace-pre-line">
                  {quizData.summary}
                </p>
              </div>
            )} */}
            <Quiz 
              questions={quizData.questions} 
              onComplete={() => setShowResult(true)}
            />
          </div>
        )}

        {showResult && (
          <Result 
            summary={quizData?.summary} 
            onRestart={handleRestart}
          />
        )}
      </div>
    </div>
  );
}