import React, { useState } from "react";
import { useScore } from "./ScoreContext";

function Quiz({ questions, onComplete }) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [userAnswers, setUserAnswers] = useState({});
  const [showExplanation, setShowExplanation] = useState(false);
  const { score, setScore } = useScore();
  const { clickSound } = useScore();

  if (!questions || questions.length === 0) {
    return (
      <div className=" bg-[#E5DECE] p-6 rounded-lg shadow-md text-center">
        <p className="text-gray-600">No questions available.</p>
      </div>
    );
  }

  const currentQuestion = questions[currentIndex];
  const isLastQuestion = currentIndex === questions.length - 1;

  const handleAnswer = (answer) => {
    clickSound.play();
    setUserAnswers({ ...userAnswers, [currentIndex]: answer });
    setShowExplanation(true);

    // Check if answer is correct for MCQ
    if (currentQuestion.type === "mcq" && answer === currentQuestion.answer_letter) {
      setScore((prev) => prev + 1);
    }
  };

  const handleNext = () => {
    clickSound.play();
    setShowExplanation(false);
    if (isLastQuestion) {
      onComplete();
    } else {
      setCurrentIndex(currentIndex + 1);
    }
  };

  // Helper function to compare answers (case-insensitive, trim whitespace)
  const compareAnswers = (userAns, correctAns) => {
    if (!userAns || !correctAns) return false;
    return userAns.trim().toLowerCase() === correctAns.trim().toLowerCase();
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
                className={`w-full text-left  rounded-full p-4 border-3  transition-all ${
                  showExplanation
                    ? letter === currentQuestion.answer_letter
                      ? "border-green-500 bg-green-50"
                      : userAnswers[currentIndex] === letter
                      ? "border-red-500 bg-red-50"
                      : "border-gray-200 bg-gray-50"
                    : "border-[#99968D] bg-[#FFFFFF] border-3 hover:border-[#040404] "
                } ${showExplanation ? "cursor-not-allowed " : "cursor-pointer"}`}
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
                  clickSound.play();
                  if (!showExplanation) {
                    setUserAnswers({ ...userAnswers, [currentIndex]: option });

                    if (compareAnswers(option, currentQuestion.answer)) {
                      setScore((prev) => prev + 1);
                    }

                    setShowExplanation(true);
                  }
                }}
                disabled={showExplanation}
                className={`w-full text-left p-4 rounded-full border-3  transition-all ${
                  showExplanation
                    ? compareAnswers(option, currentQuestion.answer)
                      ? "border-green-500 bg-green-50"
                      : userAnswers[currentIndex] === option
                      ? "border-red-500 bg-red-50"
                      : "border-gray-200 bg-gray-50"
                    : "border-[#99968D]  bg-[#FFFFFF] border-3 hover:border-[#040404]"
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
            <textarea
              placeholder="Type your answer here..."
              className="w-full border-3 border-[#99968D] bg-[#FFFFFF] p-2 rounded-full focus:ring-2  focus:border-transparent"
              value={userAnswers[currentIndex] || ""}
              disabled={showExplanation}
              onChange={(e) =>
                setUserAnswers({ ...userAnswers, [currentIndex]: e.target.value })
              }
            />
            {!showExplanation && (
              <button
                className="mt-3 bg-[#040404] text-white px-6 py-2 rounded-full hover:cursor-pointer"
                onClick={() => {clickSound.play();
                  if (compareAnswers(userAnswers[currentIndex], currentQuestion.answer)) {
                    setScore((prev) => prev + 1);
                  }
                  setShowExplanation(true);
                  clickSound.play();

                }}
              >
                Submit Answer
              </button>
            )}
            {showExplanation && (
              <div className={`mt-3 p-3 rounded-lg border-2 ${
                compareAnswers(userAnswers[currentIndex], currentQuestion.answer)
                  ? "border-green-500 bg-green-50"
                  : "border-red-500 bg-red-50"
              }`}>
                <p className="text-sm font-medium">
                  Your answer: {userAnswers[currentIndex] || "(empty)"}
                </p>
              </div>
            )}
          </div>
        );

      case "fillblank":
        return (
          <div>
            <input
            type="text"
              placeholder="Type your answer here..."
              className="w-full border-3 border-[#99968D] bg-[#FFFFFF] p-2 rounded-full focus:ring-2  focus:border-transparent"
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
                  if (compareAnswers(userAnswers[currentIndex], currentQuestion.answer)) {
                    setScore((prev) => prev + 1);
                  };
                  clickSound.play();
                  setShowExplanation(true);
                }}
                className="mt-3 bg-[#040404] text-white px-6 py-2 rounded-full hover:cursor-pointer"
              >
                Submit Answer
              </button>
            )}
            {showExplanation && (
              <div className={`mt-3 p-3 rounded-lg border-2 ${
                compareAnswers(userAnswers[currentIndex], currentQuestion.answer)
                  ? "border-green-500 bg-green-50"
                  : "border-red-500 bg-red-50"
              }`}>
                <p className="text-sm font-medium">
                  Your answer: {userAnswers[currentIndex] || "(empty)"}
                </p>
              </div>
            )}
          </div>
        );

      default:
        return <p className="text-gray-600">Unsupported question type.</p>;
    }
  };

  return (
    <div className="bg-[#E5DECE] mb-5 p-6 border-3 border-[#99968D] rounded-4xl shadow-lg">
      <div className="mb-4 flex justify-between items-center">
        <span className="text-sm font-semibold text-gray-500">
          Question {currentIndex + 1} of {questions.length}
        </span>
        <span className="text-sm font-semibold text-[#040404]">
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
        <div className="mb-4 p-4 bg-[#FFFFFF] border-3 border-[#99968D] rounded-4xl">
          <p className="font-semibold text-[#040404] mb-2">
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
          className="w-full bg-[#040404] text-[#FFFFFF] px-6 py-3 rounded-full hover:cursor-pointer font-semibold"
        >
          {isLastQuestion ? "Finish Quiz" : "Next Question"}
        </button>
      )}
    </div>
  );
}

export default Quiz;