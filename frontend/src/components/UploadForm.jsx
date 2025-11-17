import React, { useState, useEffect } from "react";

function UploadForm({ onQuizReady, setError }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [fileId, setFileId] = useState(null);
  const [statusMessage, setStatusMessage] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a file first.");
      return;
    }

    setLoading(true);
    setError("");
    setProgress(0);
    setStatusMessage("Uploading file...");

    const formData = new FormData();
    formData.append("file", file);

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
    <div className=" p-8 rounded-lg backdrop-blur-lg bg-white/10 shadow-md max-w-md mx-auto border-1 border-white/50">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-white mb-2">
            Upload Document (PDF, DOCX, TXT)
          </label>
          <input
            type="file"
            accept=".pdf,.docx,.txt"
            onChange={(e) => setFile(e.target.files[0])}
            className="w-full border border-gray-300 p-2 rounded focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            disabled={loading}
          />
        </div>

        {file && (
          <div className="text-sm text-green-600  p-2 rounded">
            Selected: {file.name}
          </div>
        )}

        <button
          type="submit"
          disabled={loading || !file}
          className="w-full bg-indigo-600 text-white px-4 py-3 rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-semibold transition-colors"
        >
          {loading ? "Processing..." : "Upload & Generate Quiz"}
        </button>
      </form>

      {loading && (
        <div className="mt-6">
          <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
            <span>{statusMessage}</span>
            <span className="font-semibold">{progress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
            <div
              className="bg-indigo-600 h-full transition-all duration-500 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default UploadForm;