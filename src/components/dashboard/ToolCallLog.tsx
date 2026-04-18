"use client";

import { motion, AnimatePresence } from "framer-motion";
import { Terminal, CheckCircle2, Clock, AlertCircle, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { ToolCall } from "@/lib/types";

interface ToolCallLogProps {
  toolCalls: ToolCall[];
  className?: string;
}

const toolIcons: Record<string, string> = {
  geological_survey_lookup: "🗺️",
  spectral_analysis: "🌈",
  terrain_classifier: "⛰️",
  proximity_search: "📡",
  risk_assessment: "🛡️",
  generate_report: "📋",
};

const statusConfig = {
  pending: {
    icon: Clock,
    color: "#64748b",
    bg: "rgba(100,116,139,0.1)",
    border: "rgba(100,116,139,0.2)",
    label: "Pending",
  },
  running: {
    icon: Loader2,
    color: "#f59e0b",
    bg: "rgba(245,158,11,0.1)",
    border: "rgba(245,158,11,0.2)",
    label: "Running",
  },
  completed: {
    icon: CheckCircle2,
    color: "#10b981",
    bg: "rgba(16,185,129,0.1)",
    border: "rgba(16,185,129,0.2)",
    label: "Complete",
  },
  error: {
    icon: AlertCircle,
    color: "#ef4444",
    bg: "rgba(239,68,68,0.1)",
    border: "rgba(239,68,68,0.2)",
    label: "Error",
  },
};

export default function ToolCallLog({ toolCalls, className = "" }: ToolCallLogProps) {
  return (
    <div className={`bg-[#1a2332] rounded-xl border border-white/[0.06] flex flex-col ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/[0.06]">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-emerald-400" />
          <h3 className="text-sm font-medium text-slate-200">Tool Call Log</h3>
        </div>
        <Badge
          variant="outline"
          className="text-[9px] border-white/10 text-slate-400"
        >
          {toolCalls.length} calls
        </Badge>
      </div>

      {/* Tool calls */}
      <ScrollArea className="flex-1" style={{ maxHeight: "280px" }}>
        <div className="p-3 space-y-2">
          {toolCalls.length === 0 ? (
            <div className="text-center py-6">
              <Terminal className="w-6 h-6 text-slate-700 mx-auto mb-2" />
              <p className="text-[10px] text-slate-600">
                Tool calls will appear here when analysis runs
              </p>
            </div>
          ) : (
            <AnimatePresence>
              {toolCalls.map((tc, index) => {
                const status = statusConfig[tc.status];
                const StatusIcon = status.icon;

                return (
                  <motion.div
                    key={tc.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.08, duration: 0.3 }}
                    className="relative"
                  >
                    {/* Connection line */}
                    {index < toolCalls.length - 1 && (
                      <div className="absolute left-[15px] top-[36px] bottom-0 w-px bg-white/[0.04]" />
                    )}

                    <div
                      className="flex items-start gap-3 p-2.5 rounded-lg hover:bg-white/[0.02] transition-colors"
                      style={{
                        borderLeft: `2px solid ${status.border}`,
                      }}
                    >
                      {/* Status indicator */}
                      <div
                        className="w-[30px] h-[30px] rounded-lg flex items-center justify-center flex-shrink-0"
                        style={{ background: status.bg }}
                      >
                        <StatusIcon
                          className={`w-3.5 h-3.5 ${
                            tc.status === "running" ? "animate-spin" : ""
                          }`}
                          style={{ color: status.color }}
                        />
                      </div>

                      {/* Content */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-base">
                            {toolIcons[tc.tool_name] || "🔧"}
                          </span>
                          <span className="text-xs font-mono font-medium text-slate-300">
                            {tc.tool_name}
                          </span>
                        </div>

                        {/* Arguments */}
                        <div className="text-[10px] text-slate-600 font-mono truncate">
                          ({JSON.stringify(tc.arguments).slice(0, 80)}
                          {JSON.stringify(tc.arguments).length > 80 ? "..." : ""})
                        </div>

                        {/* Meta */}
                        <div className="flex items-center gap-3 mt-1">
                          <span
                            className="text-[9px] font-medium uppercase tracking-wider"
                            style={{ color: status.color }}
                          >
                            {status.label}
                          </span>
                          {tc.duration_ms && (
                            <span className="text-[9px] text-slate-600">
                              {tc.duration_ms}ms
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
