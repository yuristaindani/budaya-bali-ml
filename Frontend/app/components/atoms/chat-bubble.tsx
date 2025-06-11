// import React, { useState } from "react";

// type Source = {
//   title: string;
//   url: string;
//   image?: string;
// };

// type ChatBubbleProps = {
//   isUser: boolean;
//   message: string;
//   sources?: Source[];
// };

// const ChatBubble: React.FC<ChatBubbleProps> = ({ isUser, message, sources }) => {
//   const [showSources, setShowSources] = useState(false);

//   return (
//     <div className={`w-full flex ${isUser ? "justify-end" : "justify-start"}`}>
//       <div
//         className={`w-fit max-w-lg px-5 py-3 rounded-xl 
//           ${isUser 
//             ? "rounded-br-none bg-[#f4f2ee] text-[#56411c]" 
//             : "rounded-bl-none bg-[#f5f1ec] text-[#56411c]"
//           } shadow-lg`}
//       >
//         <div className="space-y-2">
//           <p className="whitespace-pre-wrap font-sans">{message}</p>

//           {!isUser && sources && sources.length > 0 && (
//             <div className="pt-2">
//               <button
//                 onClick={() => setShowSources(!showSources)}
//                 className="flex items-center gap-2 text-sm bg-[#dedede] hover:bg-[#f7f7f7] text-[#56411c] px-3 py-1 rounded-full transition"
//               >
//                 🌐 Sumber
//               </button>

//               {showSources && (
//                 <div className="mt-2 space-y-1">
//                   {sources.map((src, idx) => (
//                     <a
//                       key={idx}
//                       href={src.url}
//                       target="_blank"
//                       rel="noopener noreferrer"
//                       className="block text-xs text-[#4495fd] hover:text-[#56411c] underline truncate"
//                     >
//                       🔗 {src.title}
//                     </a>
//                   ))}
//                 </div>
//               )}
//             </div>
//           )}
//         </div>
//       </div>
//     </div>
//   );
// };

// export default ChatBubble;

// import React, { useState } from "react";

// type Source = {
//   title: string;
//   url: string;
//   image?: string;
// };

// type ChatBubbleProps = {
//   isUser: boolean;
//   message: string;
//   sources?: Source[];
// };

// const ChatBubble: React.FC<ChatBubbleProps> = ({ isUser, message, sources }) => {
//   const [showSources, setShowSources] = useState(false);

//   // Ambil gambar pertama jika ada
//   const firstImage = sources && sources.length > 0 ? sources[0].image : undefined;

//   return (
//     <div className={`w-full flex ${isUser ? "justify-end" : "justify-start"}`}>
//       <div
//         className={`w-fit max-w-lg px-5 py-3 rounded-xl 
//           ${isUser 
//             ? "rounded-br-none bg-[#f4f2ee] text-[#56411c]" 
//             : "rounded-bl-none bg-[#f5f1ec] text-[#56411c]"
//           } shadow-lg`}
//       >
//         <div className="space-y-2">
//           <p className="whitespace-pre-wrap font-sans">{message}</p>

//           {/* Hanya tampilkan gambar pertama */}
//           {!isUser && firstImage && (
//             <div className="mt-2">
//               <img 
//                 src={firstImage}
//                 alt="Gambar referensi"
//                 className="max-w-xs rounded-lg border border-gray-200"
//               />
//             </div>
//           )}

//           {/* Tombol dan daftar sumber (tanpa gambar) */}
//           {!isUser && sources && sources.length > 0 && (
//             <div className="pt-2">
//               <button
//                 onClick={() => setShowSources(!showSources)}
//                 className="flex items-center gap-2 text-sm bg-[#dedede] hover:bg-[#f7f7f7] text-[#56411c] px-3 py-1 rounded-full transition"
//               >
//                 🌐 Sumber ({sources.length})
//               </button>

//               {showSources && (
//                 <div className="mt-2 space-y-1">
//                   {sources.map((src, idx) => (
//                     <a
//                       key={idx}
//                       href={src.url}
//                       target="_blank"
//                       rel="noopener noreferrer"
//                       className="block text-xs text-[#4495fd] hover:text-[#56411c] underline truncate"
//                     >
//                       {idx + 1}. {src.title}
//                     </a>
//                   ))}
//                 </div>
//               )}
//             </div>
//           )}
//         </div>
//       </div>
//     </div>
//   );
// };

// export default ChatBubble;

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

const ChatBubble: React.FC<ChatBubbleProps> = ({ isUser, message, sources, language }) => {
  const [showSources, setShowSources] = useState(false);
  const firstImage = sources?.[0]?.image;

  return (
    <div className={`w-full flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`w-fit max-w-lg px-5 py-3 rounded-xl 
        ${isUser ? "rounded-br-none bg-[#f4f2ee]" : "rounded-bl-none bg-[#f5f1ec]"} shadow-lg`}>
        
        {language && !isUser && (
          <span className="text-xs text-gray-500 mb-1 block">
            Detected: {language.toUpperCase()}
          </span>
        )}

        <p className="whitespace-pre-wrap font-sans">{message}</p>

        {!isUser && firstImage && (
          <div className="mt-3">
            <img 
              src={firstImage}
              alt="Reference visual"
              className="max-w-xs rounded-lg border border-gray-200"
            />
          </div>
        )}

        {!isUser && sources && sources.length > 0 && (
          <div className="pt-2">
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-2 text-sm bg-[#dedede] hover:bg-[#f7f7f7] px-3 py-1 rounded-full"
            >
              🌐 Sources ({sources.length})
            </button>

            {showSources && (
              <div className="mt-2 space-y-1">
                {sources.map((src, idx) => (
                  <a
                    key={idx}
                    href={src.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block text-xs text-[#4495fd] hover:text-[#56411c] underline truncate"
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