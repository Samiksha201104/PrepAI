// src/components/Footer.jsx
import React from "react";

export default function Footer() {
  return (
    <footer className="w-full text-center text-black py-2 mt-10 border-t border-white/40 bg-white/10 backdrop-blur-sm">
      <p className="text-sm">
        © {new Date().getFullYear()} StudyBuddy — Made for learners
      </p>
    </footer>
  );
}
