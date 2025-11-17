import React, { useState } from "react";

function Quiz({ questions, onComplete }) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [userAnswers, setUserAnswers] = useState({});
  const [showExplanation, setShowExplanation] = useState(false);
  const [score, setScore] = useState(0);

  if (!questions || questions.length === 0) {
    return (
      <div className="backdrop-blur-lg bg-white/20 p-6 rounded-lg shadow-md text-center">
        <p className="text-gray-600">No questions available.</p>
      </div>
    );
  }

  const currentQuestion = questions[currentIndex];
  const isLastQuestion = currentIndex === questions.length - 1;

  const handleAnswer = (answer) => {
    setUserAnswers({ ...userAnswers, [currentIndex]: answer });
    setShowExplanation(true);

    // Check if answer is correct for MCQ
    if (currentQuestion.type === "mcq" && answer === currentQuestion.answer_letter) {
      setScore((prev) => prev + 1);
    }
  };

  const handleNext = () => {
    setShowExplanation(false);
    if (isLastQuestion) {
      onComplete();
    } else {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const renderQuestion = () => {
    switch (currentQuestion.type) {
      case "mcq":
        return (
          <div className="space-y-3">
            {Object.entries(currentQuestion.choices || {}).map(([letter, text]) => (
              <button
                key={letter}
                onClick={() => handleAnswer(letter)}
                disabled={showExplanation}
                className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
                  showExplanation
                    ? letter === currentQuestion.answer_letter
                      ? "border-green-500 bg-green-50"
                      : userAnswers[currentIndex] === letter
                      ? "border-red-500 bg-red-50"
                      : "border-gray-200 bg-gray-50"
                    : "border-gray-300 hover:border-indigo-500 hover:bg-indigo-50"
                } ${showExplanation ? "cursor-not-allowed" : "cursor-pointer"}`}
              >
                <span className="font-semibold">{letter})</span> {text}
              </button>
            ))}
          </div>
        );

      case "tf":
        return (
          <div className="space-y-3">
            {["True", "False"].map((option) => (
              <button
                key={option}
                onClick={() => {
                  if (!showExplanation) {
                    setUserAnswers({ ...userAnswers, [currentIndex]: option });

                    if (option === currentQuestion.answer) {
                      setScore((prev) => prev + 1);
                    }

                    setShowExplanation(true);
                  }
                }}
                disabled={showExplanation}
                className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
                  showExplanation
                    ? option === currentQuestion.answer
                      ? "border-green-500 bg-green-50"
                      : userAnswers[currentIndex] === option
                      ? "border-red-500 bg-red-50"
                      : "border-gray-200 bg-gray-50"
                    : "border-gray-300 hover:border-indigo-500 hover:bg-indigo-50"
                } ${showExplanation ? "cursor-not-allowed" : "cursor-pointer"}`}
              >
                {option}
              </button>
            ))}
          </div>
        );

      case "short":
        return (
          <div>
            <p className="mb-2 font-medium">{currentQuestion.question}</p>
            <input
              type="text"
              placeholder="Type your answer here..."
              className="w-full border-2 border-gray-300 p-2 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              value={userAnswers[currentIndex] || ""}
              disabled={showExplanation}
              onChange={(e) =>
                setUserAnswers({ ...userAnswers, [currentIndex]: e.target.value })
              }
            />
            {!showExplanation && (
              <button
                className="mt-3 bg-indigo-600 text-white px-6 py-2 rounded-lg hover:bg-indigo-700"
                onClick={() => {
                  const userAnswer = userAnswers[currentIndex]?.trim().toLowerCase();
                  const correctAnswer = currentQuestion.answer.trim().toLowerCase();
                  if (userAnswer === correctAnswer) setScore((prev) => prev + 1);
                  setShowExplanation(true);
                }}
              >
                Submit Answer
              </button>
            )}
          </div>
        );

      case "fillblank":
        return (
          <div>
            <textarea
              placeholder="Type your answer here..."
              className="w-full border-2 border-gray-300 p-4 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              rows="3"
              disabled={showExplanation}
              value={userAnswers[currentIndex] || ""}
              onChange={(e) =>
                setUserAnswers({ ...userAnswers, [currentIndex]: e.target.value })
              }
            />
            {!showExplanation && (
              <button
                onClick={() => {
                  const userAnswer = userAnswers[currentIndex]?.trim().toLowerCase();
                  const correctAnswer = currentQuestion.answer.trim().toLowerCase();
                  if (userAnswer === correctAnswer) setScore((prev) => prev + 1);
                  setShowExplanation(true);
                }}
                className="mt-3 bg-indigo-600 text-white px-6 py-2 rounded-lg hover:bg-indigo-700"
              >
                Submit Answer
              </button>
            )}
          </div>
        );

      default:
        return <p className="text-gray-600">Unsupported question type.</p>;
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg">
      <div className="mb-4 flex justify-between items-center">
        <span className="text-sm font-semibold text-gray-500">
          Question {currentIndex + 1} of {questions.length}
        </span>
        <span className="text-sm font-semibold text-indigo-600">
          Score: {score}/{questions.length}
        </span>
      </div>

      <div className="mb-6">
        <h3 className="text-xl font-semibold text-gray-800 mb-4">
          {currentQuestion.question || "No question text available"}
        </h3>
        {renderQuestion()}
      </div>

      {showExplanation && (
        <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="font-semibold text-blue-900 mb-2">
            Correct Answer: {currentQuestion.answer_text || currentQuestion.answer}
          </p>
          {currentQuestion.explanation && (
            <p className="text-sm text-gray-700">{currentQuestion.explanation}</p>
          )}
        </div>
      )}

      {showExplanation && (
        <button
          onClick={handleNext}
          className="w-full bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 font-semibold"
        >
          {isLastQuestion ? "Finish Quiz" : "Next Question"}
        </button>
      )}
    </div>
  );
}

export default Quiz;
