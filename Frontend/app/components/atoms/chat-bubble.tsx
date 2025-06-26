import React, { useState } from "react";

type Source = {
  title: string;
  url: string;
  image?: string;
};

type ChatBubbleProps = {
  isUser: boolean;
  message: string;
  sources?: Source[];
  language?: string;
};

const ChatBubble: React.FC<ChatBubbleProps> = ({
  isUser,
  message,
  sources,
  language,
}) => {
  const [showSources, setShowSources] = useState(false);

  return (
    <div className={`w-full flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`w-fit max-w-lg px-5 py-3 rounded-2xl border font-montserrat
        ${
          isUser
            ? "bg-[#f4f4f4] border-[#ededed] text-[#181818]"
            : "bg-[#ffffff] border-[#ededed] text-[#000000]"
        } 
        `}
      >
        {language && !isUser && (
          <span className="text-xs text-[#888] mb-1 block font-montserrat">
            Deteksi: {language.toUpperCase()}
          </span>
        )}

        <p className="whitespace-pre-wrap">{message}</p>

        {!isUser && sources && sources.length > 0 && (
          <div className="pt-2">
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-2 text-xs bg-[#ededed] text-[#181818] px-3 py-1 rounded-full transition border border-[#d6d6d6]"
            >
              🌐 Sumber ({sources.length})
            </button>

            {showSources && (
              <div className="mt-2 space-y-1">
                {sources.map((src, idx) => (
                  <a
                    key={idx}
                    href={src.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block text-xs text-[#181818] underline truncate"
                  >
                    {idx + 1}. {src.title}
                  </a>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatBubble;