"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Sparkles, Loader2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import type { ChatMessage, ToolCall } from "@/lib/types";

interface ChatInterfaceProps {
  messages: ChatMessage[];
  isTyping: boolean;
  onSendMessage: (message: string) => void;
  className?: string;
}

function ToolCallBadge({ toolCall }: { toolCall: ToolCall }) {
  const statusIcon = {
    pending: "⏳",
    running: "🔄",
    completed: "✅",
    error: "❌",
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-white/[0.04] border border-white/[0.06] text-[10px] text-slate-400"
    >
      <span>{statusIcon[toolCall.status]}</span>
      <span className="font-mono text-emerald-400/80">{toolCall.tool_name}</span>
      {toolCall.duration_ms && (
        <span className="text-slate-600">{toolCall.duration_ms}ms</span>
      )}
    </motion.div>
  );
}

export default function ChatInterface({
  messages,
  isTyping,
  onSendMessage,
  className = "",
}: ChatInterfaceProps) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  const handleSubmit = () => {
    const trimmed = input.trim();
    if (!trimmed) return;
    onSendMessage(trimmed);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const quickQuestions = [
    "What minerals are most promising here?",
    "Assess the risk profile",
    "What exploration methods do you recommend?",
    "Compare with nearby deposits",
  ];

  return (
    <div
      className={`bg-[#1a2332] rounded-xl border border-white/[0.06] flex flex-col ${className}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/[0.06]">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
            <Bot className="w-4 h-4 text-emerald-400" />
          </div>
          <div>
            <h3 className="text-sm font-medium text-slate-200">
              AI Geologist
            </h3>
            <p className="text-[10px] text-emlate-500">
              Powered by Gemma 4
            </p>
          </div>
        </div>
        <Badge
          variant="outline"
          className="text-[9px] border-emerald-500/30 text-emerald-400 bg-emerald-500/5"
        >
          <Sparkles className="w-2.5 h-2.5 mr-1" />
          Function Calling Active
        </Badge>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1" style={{ maxHeight: "320px" }}>
        <div ref={scrollRef} className="p-4 space-y-4">
          {messages.length === 0 && (
            <div className="text-center py-8">
              <div className="w-12 h-12 rounded-2xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center mx-auto mb-3">
                <Sparkles className="w-5 h-5 text-slate-600" />
              </div>
              <p className="text-xs text-slate-500 mb-3">
                Ask about mineral prospectivity, geological context, or
                exploration strategies
              </p>
              <div className="flex flex-wrap gap-1.5 justify-center">
                {quickQuestions.map((q, i) => (
                  <button
                    key={i}
                    onClick={() => onSendMessage(q)}
                    className="px-2.5 py-1 text-[10px] text-slate-400 bg-white/[0.03] border border-white/[0.06] rounded-lg hover:bg-white/[0.06] hover:text-emerald-400 transition-all cursor-pointer"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          <AnimatePresence>
            {messages.map((msg) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="flex gap-2.5"
              >
                <div
                  className={`w-6 h-6 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5 ${
                    msg.role === "user"
                      ? "bg-slate-700/50 border border-slate-600/30"
                      : "bg-emerald-500/10 border border-emerald-500/20"
                  }`}
                >
                  {msg.role === "user" ? (
                    <User className="w-3 h-3 text-slate-400" />
                  ) : (
                    <Bot className="w-3 h-3 text-emerald-400" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-[10px] text-slate-600 mb-1">
                    {msg.role === "user" ? "You" : "Gemma 4"}
                  </div>
                  <div
                    className={`text-xs leading-relaxed ${
                      msg.role === "user"
                        ? "text-slate-300"
                        : "text-slate-400"
                    }`}
                  >
                    {msg.content.split("\n").map((line, i) => (
                      <span key={i}>
                        {line.startsWith("**") ? (
                          <strong className="text-slate-300">
                            {line.replace(/\*\*/g, "")}
                          </strong>
                        ) : line.startsWith("## ") ? (
                          <strong className="text-sm text-slate-200 block mt-1">
                            {line.replace("## ", "")}
                          </strong>
                        ) : (
                          line
                        )}
                        {i < msg.content.split("\n").length - 1 && <br />}
                      </span>
                    ))}
                  </div>
                  {msg.tool_calls && msg.tool_calls.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-2">
                      {msg.tool_calls.map((tc) => (
                        <ToolCallBadge key={tc.id} toolCall={tc} />
                      ))}
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {isTyping && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex gap-2.5"
            >
              <div className="w-6 h-6 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center flex-shrink-0">
                <Bot className="w-3 h-3 text-emerald-400" />
              </div>
              <div className="flex items-center gap-1.5 py-2">
                <div className="flex gap-1">
                  {[0, 1, 2].map((i) => (
                    <motion.div
                      key={i}
                      className="w-1.5 h-1.5 rounded-full bg-emerald-400/50"
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{
                        duration: 1,
                        repeat: Infinity,
                        delay: i * 0.2,
                      }}
                    />
                  ))}
                </div>
                <span className="text-[10px] text-slate-600">
                  Gemma 4 is thinking...
                </span>
              </div>
            </motion.div>
          )}
        </div>
      </ScrollArea>

      {/* Input */}
      <div className="p-3 border-t border-white/[0.06]">
        <div className="flex items-end gap-2 bg-white/[0.03] rounded-xl border border-white/[0.06] px-3 py-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask the AI geologist..."
            rows={1}
            className="flex-1 bg-transparent text-sm text-slate-300 placeholder-slate-600 resize-none outline-none min-h-[24px] max-h-[80px]"
            style={{
              height: "auto",
              overflow: "hidden",
            }}
            onInput={(e) => {
              const target = e.target as HTMLTextAreaElement;
              target.style.height = "auto";
              target.style.height = target.scrollHeight + "px";
            }}
          />
          <Button
            size="sm"
            onClick={handleSubmit}
            disabled={!input.trim() || isTyping}
            className="h-8 w-8 p-0 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 border-0 rounded-lg"
          >
            {isTyping ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Send className="w-3.5 h-3.5" />
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}
