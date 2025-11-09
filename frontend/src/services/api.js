// src/services/api.js
export async function uploadPdf(file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch("/api/upload", {
    method: "POST",
    body: form
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Upload failed");
  }
  return res.json(); // expects quiz JSON
}

export async function submitAnswers(quizId, answers) {
  const res = await fetch("/api/quiz/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ quizId, answers })
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Submit failed");
  }
  return res.json(); // expects {score, total, feedback}
}
