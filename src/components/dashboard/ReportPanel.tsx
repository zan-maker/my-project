"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  FileText,
  Download,
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MINERALS } from "@/lib/mock-data";
import type { ProspectivityReport, Mineral } from "@/lib/types";

interface ReportPanelProps {
  report: ProspectivityReport | null;
  isGenerating: boolean;
  onGenerate: () => void;
  className?: string;
}

function MarkdownLike({ text }: { text: string }) {
  return (
    <div className="text-xs text-slate-400 leading-relaxed space-y-2">
      {text.split("\n\n").map((para, i) => (
        <div key={i}>
          {para.split("\n").map((line, j) => {
            if (line.startsWith("## ")) {
              return (
                <h4 key={j} className="text-sm font-semibold text-slate-200 mt-3 mb-1">
                  {line.replace("## ", "")}
                </h4>
              );
            }
            if (line.startsWith("- ")) {
              return (
                <div key={j} className="flex items-start gap-2 ml-2">
                  <span className="text-emerald-400 mt-0.5">•</span>
                  <span>{line.replace("- ", "")}</span>
                </div>
              );
            }
            if (line.startsWith("**") && line.endsWith("**")) {
              return (
                <strong key={j} className="text-slate-300">
                  {line.replace(/\*\*/g, "")}
                </strong>
              );
            }
            if (line.includes("**")) {
              const parts = line.split(/(\*\*[^*]+\*\*)/g);
              return (
                <span key={j}>
                  {parts.map((part, k) =>
                    part.startsWith("**") && part.endsWith("**") ? (
                      <strong key={k} className="text-slate-300">
                        {part.replace(/\*\*/g, "")}
                      </strong>
                    ) : (
                      <span key={k}>{part}</span>
                    )
                  )}
                </span>
              );
            }
            return <span key={j}>{line}</span>;
          })}
        </div>
      ))}
    </div>
  );
}

function ReportSection({
  title,
  content,
  defaultOpen = false,
}: {
  title: string;
  content: string;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="border border-white/[0.04] rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-3 py-2.5 hover:bg-white/[0.02] transition-colors"
      >
        <span className="text-xs font-semibold text-slate-400">{title}</span>
        {open ? (
          <ChevronDown className="w-3.5 h-3.5 text-slate-500" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5 text-slate-500" />
        )}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-3 pb-3">
              <MarkdownLike text={content} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function ReportPanel({
  report,
  isGenerating,
  onGenerate,
  className = "",
}: ReportPanelProps) {
  if (!report && !isGenerating) {
    return (
      <div className={`bg-[#1a2332] rounded-xl border border-white/[0.06] p-5 ${className}`}>
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <div className="w-14 h-14 rounded-2xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center mb-3">
            <FileText className="w-6 h-6 text-slate-600" />
          </div>
          <h3 className="text-sm font-medium text-slate-400 mb-1">
            Prospectivity Report
          </h3>
          <p className="text-xs text-slate-600 max-w-[220px] mb-4">
            Run an analysis first, then generate a comprehensive prospectivity
            report
          </p>
          <Button
            onClick={onGenerate}
            disabled={true}
            className="bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 border border-emerald-500/20"
          >
            <FileText className="w-4 h-4 mr-2" />
            Generate Report
          </Button>
        </div>
      </div>
    );
  }

  if (isGenerating) {
    return (
      <div className={`bg-[#1a2332] rounded-xl border border-white/[0.06] p-5 ${className}`}>
        <div className="flex items-center gap-3 mb-4">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-sm font-medium text-slate-300">
            Generating report...
          </span>
        </div>
        <div className="space-y-3">
          {[
            "Collecting spectral analysis data",
            "Compiling terrain classification",
            "Aggregating proximity search results",
            "Evaluating risk factors",
            "Generating executive summary",
          ].map((step, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.5 }}
              className="flex items-center gap-3"
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{
                  duration: 1,
                  repeat: Infinity,
                  ease: "linear",
                  delay: i * 0.5,
                }}
                className="w-3 h-3 border border-emerald-400/30 border-t-emerald-400 rounded-full opacity-0"
                onAnimationStart={(e) => {
                  (e.target as HTMLElement).style.opacity = "1";
                }}
              />
              <span className="text-xs text-slate-500">{step}</span>
            </motion.div>
          ))}
        </div>
      </div>
    );
  }

  if (!report) return null;

  return (
    <div className={`bg-[#1a2332] rounded-xl border border-white/[0.06] flex flex-col ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/[0.06]">
        <div className="flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          <div>
            <h3 className="text-sm font-medium text-slate-200">
              Prospectivity Report
            </h3>
            <p className="text-[10px] text-slate-600">
              {new Date(report.generated_at).toLocaleString()}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Badge
            variant="outline"
            className="text-[9px] border-emerald-500/30 text-emerald-400"
          >
            <Sparkles className="w-2.5 h-2.5 mr-1" />
            {(report.confidence * 100).toFixed(0)}% confidence
          </Badge>
          <Button
            size="sm"
            variant="ghost"
            className="h-7 text-[10px] text-slate-400 hover:text-emerald-400"
          >
            <Download className="w-3 h-3 mr-1" />
            Export
          </Button>
        </div>
      </div>

      {/* Report content */}
      <ScrollArea className="flex-1" style={{ maxHeight: "500px" }}>
        <div className="p-4 space-y-3">
          {/* Target minerals */}
          <div className="flex flex-wrap gap-1.5">
            {report.mineral_targets.map((m: Mineral) => {
              const info = MINERALS[m];
              return (
                <Badge
                  key={m}
                  className="text-[10px] border-white/10 text-slate-300 bg-white/[0.03]"
                >
                  {info?.icon} {info?.name || m}
                </Badge>
              );
            })}
          </div>

          <ReportSection
            title="Executive Summary"
            content={report.executive_summary}
            defaultOpen={true}
          />
          <ReportSection
            title="Spectral Findings"
            content={report.spectral_findings}
          />
          <ReportSection
            title="Terrain Analysis"
            content={report.terrain_analysis}
          />
          <ReportSection
            title="Proximity Assessment"
            content={report.proximity_assessment}
          />
          <ReportSection
            title="Risk Evaluation"
            content={report.risk_evaluation}
          />

          {/* Recommendations */}
          <div className="bg-emerald-500/5 border border-emerald-500/10 rounded-lg p-3">
            <h4 className="text-xs font-semibold text-emerald-400 mb-2 flex items-center gap-1.5">
              <Sparkles className="w-3 h-3" />
              Recommendations
            </h4>
            <div className="space-y-1.5">
              {report.recommendations.map((rec, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2 text-xs text-slate-400"
                >
                  <span className="text-emerald-400 font-mono text-[10px] mt-0.5">
                    {i + 1}.
                  </span>
                  <span>{rec}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}
