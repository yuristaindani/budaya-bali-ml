"use client";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { SendIcon } from "lucide-react";
import React from "react";

interface ChatInputProps {
  submitHandler: (userInput: string) => void;
}

const ChatInput: React.FC<ChatInputProps> = ({ submitHandler }) => {
  const [inputValue, setInputValue] = React.useState("");

  const handleInputChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(event.target.value);
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!inputValue.trim()) return;

    submitHandler(inputValue.trim());
    setInputValue("");
  };

  return (
    <div className="w-full flex gap-4 justify-center">
      <div className="w-full">
        <form className="relative" onSubmit={handleSubmit}>
          <Textarea
            placeholder="Tanyakan apapun seputar kebudayaan & pariwisata Bali..."
            className="w-full rounded-xl bg-black border border-[#181818] text-white font-montserrat h-24 resize-none overflow-hidden pr-14 py-4 
                      focus:outline-none focus:ring-0 focus:border-transparent !focus:border-transparent !focus-visible:border-transparent"
            value={inputValue}
            onChange={handleInputChange}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
          />
          <Button
            className="absolute right-2.5 bottom-2.5 bg-[#181818] text-white rounded-full hover:bg-white transition-colors duration-300"
            type="submit"
            disabled={!inputValue}
            size="icon"
          >
            <SendIcon size={22} />
          </Button>
        </form>
      </div>
    </div>
  );
};

export default ChatInput;
