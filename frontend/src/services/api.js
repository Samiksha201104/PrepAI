// src/services/api.js
export async function uploadPdf(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("http://localhost:8000/upload", {
    method: "POST",
    body: formData
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Upload failed");
  }

  return await res.json(); // { success: true }
}

export async function fetchGeneratedQuiz() {
  const res = await fetch("http://localhost:8000/quiz");
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Could not fetch generated quiz");
  }
  return await res.json(); // backend's quiz JSON
}
