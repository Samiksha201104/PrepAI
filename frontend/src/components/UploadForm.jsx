import React, { useState, useEffect } from "react";
import { useScore } from "./ScoreContext";

function UploadForm({ onQuizReady, setError }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [fileId, setFileId] = useState(null);
  const [statusMessage, setStatusMessage] = useState("");
  const [numQuestions, setNumQuestions] = useState(10);
  const [urlInput, setUrlInput] = useState("");
  const { clickSound } = useScore();
  const [questionTypes, setQuestionTypes] = useState({
    mcq: true,
    short: true,
    fillblank: true,
    tf: true
  });

  const handleQuestionTypeChange = (type) => {
    setQuestionTypes(prev => ({
      ...prev,
      [type]: !prev[type]
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    clickSound.play();
    if (file) formData.append("file", file);
    if (urlInput.trim() !== "") formData.append("url", urlInput.trim());

    if (!file && !urlInput.trim()) {
  setError("Please upload a file OR enter a URL.");
  return;
}
    // Validate at least one question type is selected
    const selectedTypes = Object.keys(questionTypes).filter(type => questionTypes[type]);
    if (selectedTypes.length === 0) {
      setError("Please select at least one question type.");
      return;
    }

    setLoading(true);
    setError("");
    setProgress(0);
    setStatusMessage("Uploading file...");

    
    
    formData.append("num_questions", numQuestions.toString());
    formData.append("question_types", selectedTypes.join(','));

    try {
      const res = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Upload failed");
      }

      setFileId(data.file_id);
      setStatusMessage("Processing document...");
      setProgress(10);
    } catch (err) {
      setError("Upload error: " + err.message);
      setLoading(false);
    }
  };

  // Poll backend for progress
  useEffect(() => {
    if (!fileId) return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`http://localhost:5000/quiz/${fileId}`);
        const data = await res.json();

        if (data.status === "done") {
          setStatusMessage("Quiz generated successfully!");
          setProgress(100);
          onQuizReady(data.quiz);
          setLoading(false);
          clearInterval(interval);
        } else if (data.status === "running") {
          setProgress(data.progress || 0);
          setStatusMessage(`Generating quiz... ${data.progress || 0}%`);
        } else if (data.status === "error") {
          throw new Error(data.error || "Quiz generation failed");
        }
      } catch (err) {
        setError("Error: " + err.message);
        setLoading(false);
        clearInterval(interval);
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(interval);
  }, [fileId, onQuizReady, setError]);

  return (
    <div className="p-8 rounded-3xl  mb-5 bg-[#E5DDD0] shadow-md max-w-md mx-auto border-3 border-[#99968D]">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-[#040404] mb-2">
            Upload Document (PDF, DOCX, TXT)
          </label>
          <input
            type="file"
            accept=".pdf,.docx,.txt"
            onClick={()=>{clickSound.play();}}
            onChange={(e) => setFile(e.target.files[0])}
            className="w-full border-3 bg-[#FFFFFF] border-[#99968D] p-2 rounded-full focus:ring focus:ring-[#040404] focus:border-transparent"
            disabled={loading}
          />

          <input
          type="text"
          onClick={()=>{clickSound.play();}}
          placeholder="Enter URL (optional)"
          value={urlInput}
          onChange={(e) => setUrlInput(e.target.value)}
          className="url-input mt-2 border-3 bg-[#FFFFFF] border-[#99968D] w-full  p-2 rounded-full focus:ring focus:ring-[#040404] focus:border-transparent"
        />

        </div>
        
        {file && (
          <div className="text-sm text-[#040404] p-2 rounded">
            Selected: {file.name}
          </div>
        )}

        <div>
          <label className="block text-sm font-medium text-[#040404] mb-2">
            Number of Questions: {numQuestions}
          </label>
          <input
            type="range"
            min="1"
            max="50"
            value={numQuestions}
            onChange={(e) => setNumQuestions(parseInt(e.target.value))}
            disabled={loading}
            className="w-full h-1 bg-[#99968D] rounded-lg appearance-none cursor-pointer accent-[#040404]"
          />
          <div className="flex justify-between text-xs text-[#040404] mt-1">
            <span>1</span>
            <span>25</span>
            <span>50</span>
          </div>
        </div>

        <div className="bg-[#FFFFFF] border-[#99968D] rounded-2xl p-2 border-3">
          <label className="block text-sm  font-medium text-[#040404] mb-3">
            Question Types
          </label>
          <div className="grid grid-cols-2 gap-2">
            <label className="flex items-center space-x-2 p-2 bg-white/20 rounded-lg cursor-pointer hover:bg-white/30 transition">
              <input
                type="checkbox"
                checked={questionTypes.mcq}
                onChange={() => handleQuestionTypeChange('mcq')}
                disabled={loading}
                className="w-4 h-4 accent-[#040404] rounded"
              />
              <span className="text-sm text-[#040404]">Multiple Choice</span>
            </label>
            <label className="flex items-center space-x-2 p-2 bg-white/20 rounded-lg cursor-pointer hover:bg-white/30 transition">
              <input
                type="checkbox"
                checked={questionTypes.short}
                // defaultChecked={false}
                onChange={() => handleQuestionTypeChange('short')}
                disabled={loading}
                className="w-4 h-4 accent-[#040404] rounded"
              />
              <span className="text-sm text-[#040404]">Short Answer</span>
            </label>
            <label className="flex items-center space-x-2 p-2 bg-white/20 rounded-lg cursor-pointer hover:bg-white/30 transition">
              <input
                type="checkbox"
                checked={questionTypes.fillblank}
                onChange={() => handleQuestionTypeChange('fillblank')}
                disabled={loading}
                className="w-4 h-4 accent-[#040404] rounded"
              />
              <span className="text-sm text-[#040404]">Fill in Blank</span>
            </label>
            <label className="flex items-center space-x-2 p-2 bg-white/20 rounded-lg cursor-pointer hover:bg-white/30 transition">
              <input
                type="checkbox"
                checked={questionTypes.tf}
                onChange={() => handleQuestionTypeChange('tf')}
                disabled={loading}
                className="w-4 h-4 accent-[#040404] rounded"
              />
              <span className="text-sm text-[#040404]">True/False</span>
            </label>
          </div>
        </div>

        <button
          type="submit"

          disabled={loading || (!file && !urlInput.trim())}
          className="w-full bg-[#040404] text-[#FFFFFF] px-4 py-3 rounded-full hover:cursor-pointer disabled:bg-[#99968D] disabled:text-[#040404] disabled:cursor-not-allowed font-semibold transition-colors"
        >
          {loading ? "Processing..." : "Upload & Generate Quiz"}
        </button>
      </form>

      {loading && (
        <div className="mt-6">
          <div className="flex items-center justify-between text-sm text-[#040404] mb-2">
            <span>{statusMessage}</span>
            <span className="font-semibold">{progress}%</span>
          </div>
          <div className="w-full bg-[#99968D] rounded-full h-3 overflow-hidden">
            <div
              className="bg-[#040404] h-full transition-all duration-500 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default UploadForm;