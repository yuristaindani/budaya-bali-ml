"use client";
import React from "react";

type ChatHistory = {
  id: string;
  title: string;
  lastMessage: string;
  createdAt: string;
};

interface ChatSidebarProps {
  histories: ChatHistory[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  sidebarOpen: boolean;
  closeSidebar: () => void;
}

const SIDEBAR_WIDTH = 260;

const ChatSidebar: React.FC<ChatSidebarProps> = ({
  histories,
  activeId,
  onSelect,
  onNewChat,
  sidebarOpen,
  closeSidebar,
}) => (
  <>
    {/* Overlay for mobile */}
    <div
      className={`fixed inset-0 bg-black/40 z-40 transition-opacity duration-200 ${
        sidebarOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
      }`}
      onClick={closeSidebar}
      aria-hidden={!sidebarOpen}
    />
    <aside
      className={`
        fixed top-16 right-0 h-[calc(100vh-4rem)] max-h-screen  border-l border-gray-600 z-50 flex flex-col pt-3
        w-[90vw] max-w-[${SIDEBAR_WIDTH}px]
        transition-transform duration-300
        ${sidebarOpen ? 'translate-x-0' : 'translate-x-full'}
      `}
      style={{
        width: 'min(90vw, 260px)',
        minHeight: 'calc(100vh - 4rem)',
      }}
    >

      {/* Obrolan Baru dengan Logo */}
      <div className="border-b border-gray-600">
        <button
          onClick={() => {
            onNewChat();
            closeSidebar();
          }}
          className="w-full text-left px-6 py-3 hover:bg-gray-600 focus:bg-gray-600 flex items-start gap-3"
        >
          <div className="bg-[#181818] text-white p-2 rounded-lg flex-shrink-0">
            <svg 
              xmlns="http://www.w3.org/2000/svg" 
              className="h-4 w-4" 
              fill="none" 
              viewBox="0 0 24 24" 
              stroke="currentColor"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={2} 
                d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" 
              />
            </svg>
          </div>
          <div>
            <div className="font-bold">Obrolan Baru</div>
            <div className="text-xs text-[#888]">Mulai percakapan baru</div>
          </div>
        </button>
      </div>

      <div className="mb-3" />
      <div className="px-5 text-xs font-bold uppercase text-[#888] tracking-wider mb-2">
        Riwayat Obrolan
      </div>
      <nav className="flex-1 overflow-y-auto">
        {histories.length === 0 && (
          <p className="px-6 text-[#aaa] text-sm mt-8">Belum ada riwayat.</p>
        )}
        <ul>
          {histories.map(chat => (
            <li key={chat.id}>
              <button
                onClick={() => {
                  onSelect(chat.id);
                  closeSidebar();
                }}
                className={`w-full text-left px-6 py-3 border-b border-gray-600 hover:bg-gray-700 focus:bg-[#f4f4f4] ${
                  activeId === chat.id ? "bg-gray-700 font-bold" : ""
                }`}
              >
                <div className="truncate">{chat.title || "Chat tanpa judul"}</div>
                <div className="text-xs text-[#888] truncate">{chat.lastMessage}</div>
                <div className="text-[10px] text-[#bbb]">{chat.createdAt}</div>
              </button>
            </li>
          ))}
        </ul>
      </nav>
    </aside>
  </>
);

export default ChatSidebar;