"use client";

import { useEffect, useRef, useState } from "react";
import { askQuestion, AskResponse } from "../../lib/api";

interface Message {
  role:     "user" | "assistant";
  content:  string;
  response?: AskResponse;
}

interface Props {
  prefillQuestion: string;         // from AutoQA chip click
  onNewResponse: (r: AskResponse) => void;  // sends latest response to ModelCompare
}

export default function ChatBox({ prefillQuestion, onNewResponse }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput]       = useState("");
  const [loading, setLoading]   = useState(false);
  const bottomRef               = useRef<HTMLDivElement>(null);

  // When a suggested question is clicked, prefill the input
  useEffect(() => {
    if (prefillQuestion) setInput(prefillQuestion);
  }, [prefillQuestion]);

  // Auto scroll to bottom on new message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleAsk = async () => {
    const q = input.trim();
    if (!q || loading) return;

    setInput("");
    setLoading(true);

    // Add user message immediately
    setMessages(prev => [...prev, { role: "user", content: q }]);

    try {
      const data = await askQuestion(q);
      setMessages(prev => [...prev, {
        role:     "assistant",
        content:  data.distilbert.answer,   // show DistilBERT answer in chat bubble
        response: data,
      }]);
      onNewResponse(data);
    } catch (err: any) {
      setMessages(prev => [...prev, {
        role:    "assistant",
        content: err.message || "Something went wrong.",
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  return (
    <div className="w-full max-w-xl mx-auto mt-6 flex flex-col">

      {/* Message history */}
      <div className="flex flex-col gap-3 min-h-[200px] max-h-[400px] overflow-y-auto
                      border border-gray-200 rounded-xl p-4 bg-gray-50">
        {messages.length === 0 && (
          <p className="text-gray-400 text-sm text-center mt-8">
            Ask a question about your document
          </p>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] px-4 py-2 rounded-2xl text-sm
                ${msg.role === "user"
                  ? "bg-blue-500 text-white rounded-br-sm"
                  : "bg-white border border-gray-200 text-gray-700 rounded-bl-sm"
                }`}
            >
              {msg.content}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-white border border-gray-200 px-4 py-2 rounded-2xl
                            rounded-bl-sm text-sm text-gray-400">
              Thinking...
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="flex gap-2 mt-3">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question..."
          disabled={loading}
          className="flex-1 border border-gray-300 rounded-xl px-4 py-2 text-sm
                     focus:outline-none focus:border-blue-400 disabled:opacity-50"
        />
        <button
          onClick={handleAsk}
          disabled={loading || !input.trim()}
          className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white text-sm
                     rounded-xl disabled:opacity-40 transition-colors"
        >
          Ask
        </button>
      </div>

    </div>
  );
}