"use client";

import { postChat } from "@/app/actions/post-chat";
import React, { useEffect, useRef, useState } from "react";
import ChatBubble from "../atoms/chat-bubble";
import ChatInput from "../molecules/message-input";
import ChatSidebar from "./chat-sidebar";
import ChatHeader from "./chat-header";
// import ChatFooter from "./chat-footer";

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

const HEADER_HEIGHT = 64; // px
const INPUT_HEIGHT = 96; // px
const EXTRA_BOTTOM_SPACE = 16;
const SIDEBAR_WIDTH = 260; // px
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
    if (typeof window !== "undefined") {
      const data = localStorage.getItem(STORAGE_KEY);
      const lastId = localStorage.getItem("budayabali_active_chat_id");

      if (data) {
        const parsed: ChatHistory[] = JSON.parse(data);
        setHistories(parsed);

        if (lastId && parsed.some(h => h.id === lastId)) {
          // Jika ada ID terakhir dan masih valid, gunakan itu
          setActiveId(lastId);
        } else {
          // Jika tidak ada ID terakhir, tetap buat obrolan baru
          handleNewChat();
        }
      } else {
        // Pertama kali buka web
        handleNewChat();
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Save histories and activeId to localStorage every change
  useEffect(() => {
    if (typeof window !== "undefined") {
      const data = localStorage.getItem(STORAGE_KEY);
      const parsed: ChatHistory[] = data ? JSON.parse(data) : [];
    }
  }, [histories, activeId]);

  // Scroll to bottom on new message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeId, histories, isPending]);

  // Create new chat
  function handleNewChat() {
    const id = crypto.randomUUID();
    const now = new Date();
    const newHistory: ChatHistory = {
      id,
      title: "Obrolan Baru",
      lastMessage: "",
      createdAt: now.toLocaleString("id-ID", { hour12: false }),
      messages: [],
    };
    setHistories(prev => {
      const updated = [newHistory, ...prev];
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated)); // simpan ke localStorage
      return updated;
    });

    setActiveId(id);
    localStorage.setItem("budayabali_active_chat_id", id); // simpan ID aktif
    setSidebarOpen(false);
  }

  function handleSendMessage(newMessage: MessageProps) {
  setHistories(prev => {
    const updated = prev.map(history => {
      if (history.id === activeId) {
        const updatedMessages = [...history.messages, newMessage];
        return {
          ...history,
          messages: updatedMessages,
          lastMessage: newMessage.content,
        };
      }
      return history;
    });
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    return updated;
  });
}

function createChatTitle(input: string) {
  // Potong jika terlalu panjang
  const maxLength = 50;
  const trimmed = input.length > maxLength 
    ? input.substring(0, maxLength) + '...' 
    : input;
  
  // Hapus karakter newline dan multiple spaces
  return trimmed.replace(/\n/g, ' ').replace(/\s+/g, ' ').trim();
}

  // Select a history
  function handleSelectHistory(id: string) {
    setActiveId(id);
    setSidebarOpen(false);
  }

  function handleDeleteHistory(id: string) {
    setHistories(prev => prev.filter(h => h.id !== id));
  
    // If the deleted chat was the active one, create a new chat
    if (activeId === id) {
      handleNewChat();
    }
  }

  // Send chat
  function handleSubmit(input: string) {
    const activeChat = histories.find(h => h.id === activeId);
    if (!activeChat) return;
    const userMessage: MessageProps = { role: "user", content: input };
    const currId = activeChat.id;

    setHistories(prev =>
      prev.map(h =>
        h.id === currId
          ? {
              ...h,
              messages: [...h.messages, userMessage],
              lastMessage: input,
              title: h.title === "Obrolan Baru" && h.messages.length === 0 ? createChatTitle(input) : h.title,
            }
          : h
      )
    );

    startTransition(async () => {
      const response = await postChat(input);
      const botMsg: MessageProps = {
        role: "assistant",
        content:
          response.success && response.data
            ? response.data.content!
            : "Maaf, terjadi kesalahan.",
        sources: response.success && response.data ? response.data.sources : undefined,
      };
      setHistories(prev =>
        prev.map(h =>
          h.id === currId
            ? {
                ...h,
                messages: [...h.messages, botMsg],
                lastMessage: botMsg.content,
              }
            : h
        )
      );
    });
  }

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
        className={`
          flex flex-col items-center transition-all duration-300
          pt-16 pb-16
          ${sidebarOpen ? 'lg:pr-[260px]' : ''}
        `}
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
          {!activeId ||
          !histories.find(h => h.id === activeId)?.messages.length ? (
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
              {histories
                .find(h => h.id === activeId)!
                .messages.map((chat, index) => (
                  <React.Fragment key={index}>
                    <ChatBubble
                      isUser={chat.role === "user"}
                      message={chat.content}
                      sources={
                        chat.role === "assistant" ? chat.sources : undefined
                      }
                    />
                    {chat.role === "assistant" &&
                      chat.sources &&
                      chat.sources.length > 0 && (
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
        {/* Fixed Input (always at bottom, width follows main content) */}
        <div
          className={`
            fixed bottom-15 left-0 w-full z-40
            transition-all duration-300
            ${sidebarOpen ? "lg:pr-[260px]" : ""}
          `}
        >
          <div className="max-w-2xl mx-auto">
            <ChatInput submitHandler={handleSubmit} />
          </div>
        </div>
      </main>
      {/* <ChatFooter /> */}
    </div>
  );
};

export default ChatBody;