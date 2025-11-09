
import React from "react";

export default function Navbar() {
  return (
    <nav className="w-full bg-white/20 backdrop-blur-md border-b border-white/40 text-white py-3 px-6 flex justify-between items-center">
      <h1 className="text-2xl font-bold">PrepAi</h1>
      <div className="flex gap-6 text-xl">
        <a href="#" className="hover:text-blue-400 transition">Home</a>
        <a href="#" className="hover:text-blue-400 transition">Contact</a>
        <a href="#" className="hover:text-blue-400 transition">About</a>
      </div>
    </nav>
  );
}
