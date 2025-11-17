import React, { useState,useEffect } from "react";
import UploadForm from "./components/UploadForm";
import Quiz from "./components/Quiz";
import Result from "./components/Result";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import sampleQuiz from "./assets/quiz4.json"
// import sampleQuiz from "./assets/quiz.json"
// import sampleQuiz from "./assets/quiz2.json"

export default function App() {  
  const [quizData, setQuizData] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(null);

 useEffect(() => {
  setQuizData({
    questions: sampleQuiz.map((q, index) => ({
      id: `q${index + 1}`,
      text: q.question,
      type:
        q.type === "mcq" ? "mcq" :
        q.type === "short" ? "open" :
        q.type === "fillblank" ? "open" :
        q.type === "tf" ? "mcq" : "open",
      options:
        q.type === "mcq"
          ? Object.values(q.choices)
          : q.type === "tf"
          ? ["True", "False"]
          : [],      
      answer: q.answer_text || q.answer || q.answer_letter, 
      correctLetter: q.answer_letter || null,
      correctText: q.answer_text || null
    }))
  });
}, []);



  return (
    <div className="min-h-screen h-full flex flex-col items-center p-6 bg-[url('/src/assets/bg.jpg')] bg-cover  ">
      <Navbar></Navbar>
      <div className={quizData?"mt-27 mb-15 w-full flex items-center justify-center":"mt-40 mb-35 w-full flex items-center justify-center"}>
        <div className="w-[100%] max-w-3xl max-h-[400px] overflow-y-auto bg-white/10 backdrop-blur-sm border border-white/50 rounded-lg shadow-md p-6 ">
          <h1 className="text-4xl text-center text-black font-semibold mb-4">PrepAI — Upload PDF, get a quiz</h1>

          {/* for showing the first part to upload */}
          {!quizData && !loading && (
            <UploadForm
              onStart={() => {setLoading(true); setResult(null)}}
              onComplete={(data) => {setQuizData(data); setLoading(false);}}
              onError={() => setLoading(false)}
            />
          )}

          {/* for the loading circle  */}
          {loading?(
            <div className="text-center py-12">
              <div className="animate-spin inline-block w-10 h-10 border-4 border-slate-300 border-t-slate-600 rounded-full mb-4" />
              <div className="text-md text-black">Generating quiz — this can take a few seconds...</div>
            </div>):null
          }

          {/* quiz has been generated but result not yet obtained */}
          {quizData && !result && (
            <Quiz
              quiz={quizData}
              onSubmitResult={(res) => setResult(res)}
              onBack={() => { setQuizData(null); setResult(null); }}
            />
          )}

          {/* result has been obtained */}
          {result && (
            <Result result={result} onRestart={() => { setQuizData(null); setResult(null); }} />
          )}
        </div>
      </div>
      <Footer></Footer>
    </div>
  );
}
