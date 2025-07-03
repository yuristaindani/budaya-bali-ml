"use client";

import { postChat } from "@/app/actions/post-chat";
import React, { useEffect, useRef, useState } from "react";
import ChatBubble from "../atoms/chat-bubble";
import ChatInput from "../molecules/message-input";
import ChatSidebar from "./chat-sidebar";
import ChatHeader from "./chat-header";

type Source = {
  title: string;
  url: string;
  image?: string;
};

type MessageProps = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

type ChatHistory = {
  id: string;
  title: string;
  lastMessage: string;
  createdAt: string;
  messages: MessageProps[];
};

const HEADER_HEIGHT = 64;
const INPUT_HEIGHT = 96;
const EXTRA_BOTTOM_SPACE = 16;
const SIDEBAR_WIDTH = 260;
const STORAGE_KEY = "budayabali_chat_histories";

function getNowString() {
  return new Date().toLocaleString("id-ID", { hour12: false });
}

const ChatBody = () => {
  const [histories, setHistories] = useState<ChatHistory[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [isPending, startTransition] = React.useTransition();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Load histories from localStorage on mount
  useEffect(() => {
    const loadFromStorage = () => {
      try {
        const data = localStorage.getItem(STORAGE_KEY);
        
        // Selalu buat new chat terlebih dahulu
        const newChatId = crypto.randomUUID();
        const now = getNowString();
        const newHistory: ChatHistory = {
          id: newChatId,
          title: "Obrolan Baru",
          lastMessage: "",
          createdAt: now,
          messages: [],
        };

        setActiveId(newChatId);
        
        // Jika ada history sebelumnya, gabungkan dengan new chat
        if (data) {
          const parsed: ChatHistory[] = JSON.parse(data);
          setHistories([newHistory, ...parsed]);
        } else {
          setHistories([newHistory]);
        }
      } catch (e) {
        console.error("Failed to load chat history", e);
        handleNewChat();
      }
    };

    loadFromStorage();
  }, []);

  // Save to localStorage whenever histories changes
  useEffect(() => {
    if (histories.length > 0) {
      // Simpan semua history kecuali new chat yang kosong
      const historiesToSave = histories.filter(h => h.messages.length > 0);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(historiesToSave));
    }
  }, [histories]);

  // Scroll to bottom on new message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeId, histories, isPending]);

  // Create new chat
  function handleNewChat() {
    const id = crypto.randomUUID();
    const now = getNowString();
    const newHistory: ChatHistory = {
      id,
      title: "Obrolan Baru",
      lastMessage: "",
      createdAt: now,
      messages: [],
    };

    setHistories(prev => {
      // Hapus new chat sebelumnya jika ada
      const filtered = prev.filter(h => h.messages.length > 0);
      return [newHistory, ...filtered];
    });

    setActiveId(id);
    setSidebarOpen(false);
  }

  function createChatTitle(input: string) {
    try {
      const maxLength = 50;
      const trimmed = input.length > maxLength 
        ? input.substring(0, maxLength) + '...' 
        : input;
      return trimmed.replace(/\n/g, ' ').replace(/\s+/g, ' ').trim();
    } catch (e) {
      console.error("Error creating chat title", e);
      return "Obrolan Baru";
    }
  }

  // Select a history
  function handleSelectHistory(id: string) {
    setActiveId(id);
    setSidebarOpen(false);
  }

  function handleDeleteHistory(id: string) {
    setHistories(prev => prev.filter(h => h.id !== id));
    if (activeId === id) {
      handleNewChat();
    }
  }

  // Send chat
  function handleSubmit(input: string) {
    if (!activeId) return;
    
    const userMessage: MessageProps = { role: "user", content: input };
    
    setHistories(prev => {
      return prev.map(history => {
        if (history.id === activeId) {
          const isFirstMessage = history.messages.length === 0;
          return {
            ...history,
            messages: [...history.messages, userMessage],
            lastMessage: input,
            title: isFirstMessage ? createChatTitle(input) : history.title,
          };
        }
        return history;
      });
    });

    startTransition(async () => {
      const response = await postChat(input);
      const botMsg: MessageProps = {
        role: "assistant",
        content: response.success && response.data
          ? response.data.content!
          : "Maaf, terjadi kesalahan.",
        sources: response.success && response.data ? response.data.sources : undefined,
      };

      setHistories(prev => {
        return prev.map(history => {
          if (history.id === activeId) {
            return {
              ...history,
              messages: [...history.messages, botMsg],
              lastMessage: botMsg.content,
            };
          }
          return history;
        });
      });
    });
  }

  const activeMessages = activeId 
    ? histories.find(h => h.id === activeId)?.messages || []
    : [];

  return (
    <div className="min-h-screen bg-[#ffffff] relative">
      {/* Header */}
      <ChatHeader onSidebarToggle={() => setSidebarOpen(s => !s)} />
      
      {/* Sidebar */}
      <ChatSidebar
        histories={histories.map(h => ({
          id: h.id,
          title: h.title,
          lastMessage: h.lastMessage,
          createdAt: h.createdAt,
        }))}
        activeId={activeId}
        onSelect={handleSelectHistory}
        onNewChat={handleNewChat}
        sidebarOpen={sidebarOpen}
        closeSidebar={() => setSidebarOpen(false)}
      />
      
      {/* Main content */}
      <main
        className={`flex flex-col items-center transition-all duration-300 pt-16 pb-16 ${
          sidebarOpen ? 'lg:pr-[260px]' : ''
        }`}
        style={{
          minHeight: "100vh",
          width: "100%",
        }}
      >
        <div
          className="w-full max-w-2xl px-4 py-6 overflow-y-auto"
          style={{
            flexGrow: 1,
            maxHeight: `calc(100vh - ${HEADER_HEIGHT}px - ${INPUT_HEIGHT}px - ${EXTRA_BOTTOM_SPACE}px)`
          }}
        >
          {activeMessages.length === 0 ? (
            <div className="text-center mt-28">
              <h1 className="text-2xl font-bold mb-4 text-[#181818]">
                Rahajeng Semeton ❀
              </h1>
              <p className="text-base max-w-xl mx-auto mb-10 text-[#444]">
                Apa yang bisa saya bantu?
              </p>
            </div>
          ) : (
            <div className="flex flex-col gap-6">
              {activeMessages.map((chat, index) => (
                <React.Fragment key={index}>
                  <ChatBubble
                    isUser={chat.role === "user"}
                    message={chat.content}
                    sources={chat.role === "assistant" ? chat.sources : undefined}
                  />
                  {chat.role === "assistant" && chat.sources && chat.sources.length > 0 && (
                    <div className="flex flex-col gap-4">
                      {chat.sources.map((src, imgIndex) =>
                        src.image ? (
                          <div
                            key={imgIndex}
                            className="w-full flex justify-start"
                          >
                            <div className="w-fit max-w-xs rounded-xl overflow-hidden shadow-md border border-[#ededed]">
                              <img
                                src={src.image}
                                alt={src.title}
                                className="w-full h-auto object-cover"
                              />
                            </div>
                          </div>
                        ) : null
                      )}
                    </div>
                  )}
                </React.Fragment>
              ))}
              {isPending && (
                <ChatBubble
                  message="Sedang menyiapkan jawaban..."
                  isUser={false}
                />
              )}
              <div ref={bottomRef} />
            </div>
          )}
        </div>
        
        {/* Fixed Input */}
        <div
          className={`fixed bottom-15 left-0 w-full z-40 transition-all duration-300 ${
            sidebarOpen ? "lg:pr-[260px]" : ""
          }`}
        >
          <div className="max-w-2xl mx-auto">
            <ChatInput submitHandler={handleSubmit} />
          </div>
        </div>
      </main>
    </div>
  );
};

export default ChatBody;