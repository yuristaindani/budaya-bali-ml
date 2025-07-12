"use client";
import React from "react";


interface ChatHeaderProps {
  onSidebarToggle: () => void;
}

const ChatHeader: React.FC<ChatHeaderProps> = ({ onSidebarToggle }) => (
  <header className="fixed top-0 left-0 w-full z-30 bg-transparent text-white border-b border-[#222] h-16 flex items-center px-8 justify-between">
    <div className="flex items-center gap-3">
      <img src="/logo-budayabali.png" alt="Logo Budaya Bali" className="h-12 w-auto" />
    </div>
    <button
      aria-label="Tampilkan sidebar"
      onClick={onSidebarToggle}
      className="p-2 rounded hover:bg-[#222] transition ml-auto z-50"
      type="button"
    >
      {/* Hamburger icon */}
      <svg width="28" height="28" fill="none" viewBox="0 0 24 24">
        <rect x="4" y="6" width="16" height="2" rx="1" fill="white"/>
        <rect x="4" y="11" width="16" height="2" rx="1" fill="white"/>
        <rect x="4" y="16" width="16" height="2" rx="1" fill="white"/>
      </svg>
    </button>
  </header>
);

export default ChatHeader;