import React, { useState } from "react";
import { uploadPdf } from "../services/api";

export default function UploadForm({ onStart, onComplete, onError }) {
  const [file, setFile] = useState(null);
  const [err, setErr] = useState("");

  function handleFile(e) 
  {
    setErr("");
    const f = e.target.files[0];
    if (!f) return;
    if (f.type !== "application/pdf")
    {
      setErr("Please upload a PDF file.");
      setFile(null);
      return;
    }
    if (f.size > 15 * 1024 * 1024) 
    { 
      setErr("File too large (max 15MB).");//limit checking
      setFile(null);
      return;
    }
    setFile(f);
  }

  async function handleSubmit(e) 
  {
    e.preventDefault();
    if (!file) { setErr("Select a PDF first."); return; }
    try {
      onStart && onStart();
      const quiz = await uploadPdf(file); // call backend
      onComplete && onComplete(quiz);
    } catch (e) {
      console.error(e);
      setErr(e.message || "Upload failed");
      onError && onError(e);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <label className="block">
        <span className="text-lg font-medium text-slate-400">Upload PDF</span>
        <input
          type="file"
          accept="application/pdf"
          onChange={handleFile} //just the change is being recorded so if it is a word it just displays error
          className="mt-2 block w-full text-md text-slate-400 file:mr-4 file:py-2 file:px-4
                     file:rounded file:border-1 file:text-md file:font-semibold
                     file:bg-black/50 file:text-white file:border-white/30 hover:file:bg-slate-800"
        />
      </label>
      {file && <div className="text-sm text-slate-400">Selected: {file.name}</div>}
      {err && <div className="text-md text-red-600">{err}</div>}

      <div className="flex gap-2">
        <button
          type="submit"
          className="px-4 py-2 bg-black/50 text-white rounded hover:bg-slate-800"
        >
          Upload & Generate Quiz
        </button>
        <button
          type="button"
          onClick={() => { setFile(null); setErr(""); }}
          className="px-4 py-2 border rounded text-white hover:bg-slate-600"
        >
          Reset
        </button>
      </div>
    </form>
  );
}
