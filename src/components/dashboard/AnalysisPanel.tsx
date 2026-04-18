"use client";

import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  Mountain,
  Radar,
  ShieldAlert,
  TrendingUp,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { useState } from "react";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MINERALS } from "@/lib/mock-data";
import type { AnalysisResult, Mineral } from "@/lib/types";

interface AnalysisPanelProps {
  result: AnalysisResult | null;
  isLoading: boolean;
  className?: string;
}

function SpectralSection({ data }: { data: AnalysisResult["spectral"] }) {
  return (
    <div className="space-y-3">
      <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
        <Activity className="w-3.5 h-3.5 text-emerald-400" />
        Spectral Analysis
      </h4>
      <div className="space-y-2">
        {data.map((s) => {
          const mineral = MINERALS[s.mineral as Mineral];
          return (
            <div key={s.mineral} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300">
                  {mineral?.icon} {mineral?.name || s.mineral}
                </span>
                <span
                  className="font-mono font-bold"
                  style={{ color: mineral?.color || "#10b981" }}
                >
                  {(s.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${s.confidence * 100}%` }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                  className="h-full rounded-full"
                  style={{ background: mineral?.color || "#10b981" }}
                />
              </div>
              <div className="text-[10px] text-slate-500">
                Anomaly score: {s.anomaly_score.toFixed(3)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TerrainSection({ data }: { data: AnalysisResult["terrain"] }) {
  return (
    <div className="space-y-3">
      <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
        <Mountain className="w-3.5 h-3.5 text-emerald-400" />
        Terrain Classification
      </h4>
      <div className="bg-white/[0.03] rounded-lg p-3 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-slate-200">{data.terrain_type}</span>
          <Badge
            variant="outline"
            className="text-[10px] border-emerald-500/30 text-emerald-400"
          >
            {data.formation}
          </Badge>
        </div>
        <p className="text-xs text-slate-400 leading-relaxed">{data.description}</p>
        <div className="flex gap-4 text-[10px] text-slate-500">
          <span>Age: {data.geological_age}</span>
          <span>
            Elev: {data.elevation_range.min}–{data.elevation_range.max}m
          </span>
        </div>
        <div className="flex flex-wrap gap-1">
          {data.features.map((f, i) => (
            <Badge
              key={i}
              variant="secondary"
              className="text-[9px] bg-white/5 text-slate-400 border-white/5"
            >
              {f}
            </Badge>
          ))}
        </div>
      </div>
    </div>
  );
}

function ProximitySection({ data }: { data: AnalysisResult["proximity"] }) {
  return (
    <div className="space-y-3">
      <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
        <Radar className="w-3.5 h-3.5 text-emerald-400" />
        Proximity Search
        <Badge variant="outline" className="text-[9px] border-white/10 text-slate-400">
          {data.deposits_found} found
        </Badge>
      </h4>
      <div className="space-y-1.5">
        {data.deposits.slice(0, 5).map((d, i) => {
          const mineral = MINERALS[d.mineral_type as Mineral];
          return (
            <div
              key={i}
              className="flex items-center justify-between py-1.5 px-2 rounded-md bg-white/[0.02] hover:bg-white/[0.05] transition-colors"
            >
              <div className="flex items-center gap-2 min-w-0">
                <div
                  className="w-2 h-2 rounded-full flex-shrink-0"
                  style={{ background: mineral?.color || "#10b981" }}
                />
                <span className="text-xs text-slate-300 truncate">{d.name}</span>
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                <Badge
                  variant="outline"
                  className="text-[9px] border-white/5 text-slate-500"
                >
                  {d.status}
                </Badge>
                <span className="text-[10px] font-mono text-slate-400">
                  {d.distance_km}km
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function RiskSection({ data }: { data: AnalysisResult["risk"] }) {
  const levelColor = {
    low: "#10b981",
    medium: "#f59e0b",
    high: "#ef4444",
    critical: "#dc2626",
  }[data.level];

  return (
    <div className="space-y-3">
      <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
        <ShieldAlert className="w-3.5 h-3.5 text-emerald-400" />
        Risk Assessment
      </h4>
      <div className="bg-white/[0.03] rounded-lg p-3">
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm text-slate-300">Overall Risk</span>
          <div className="flex items-center gap-2">
            <span
              className="text-lg font-bold font-mono"
              style={{ color: levelColor }}
            >
              {data.overall_risk}
            </span>
            <span className="text-[10px] text-slate-500">/100</span>
          </div>
        </div>
        <Progress
          value={data.overall_risk}
          className="h-2 bg-white/5"
        />
        <div className="mt-3 space-y-2">
          {data.factors.map((f) => (
            <div key={f.factor} className="flex items-center justify-between">
              <span className="text-[11px] text-slate-400">{f.factor}</span>
              <div className="flex items-center gap-2">
                <div className="w-16 h-1 bg-white/5 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${f.score}%`,
                      background:
                        f.score > 70
                          ? "#ef4444"
                          : f.score > 50
                          ? "#f59e0b"
                          : "#10b981",
                    }}
                  />
                </div>
                <span className="text-[10px] font-mono text-slate-500 w-6 text-right">
                  {f.score}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function AnalysisPanel({
  result,
  isLoading,
  className = "",
}: AnalysisPanelProps) {
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    spectral: true,
    terrain: true,
    proximity: true,
    risk: true,
  });

  const toggleSection = (key: string) => {
    setExpandedSections((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  if (isLoading) {
    return (
      <div className={`bg-[#1a2332] rounded-xl border border-white/[0.06] p-5 ${className}`}>
        <div className="flex items-center gap-3 mb-4">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-sm font-medium text-slate-300">
            Running analysis pipeline...
          </span>
        </div>
        <div className="space-y-4">
          {["Spectral Analysis", "Terrain Classification", "Proximity Search", "Risk Assessment"].map(
            (label, i) => (
              <div key={i} className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-500">{label}</span>
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    className="w-3 h-3 border border-emerald-400/30 border-t-emerald-400 rounded-full"
                  />
                </div>
                <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-emerald-400/30 rounded-full"
                    initial={{ width: "0%" }}
                    animate={{ width: "100%" }}
                    transition={{ duration: 3, delay: i * 0.5, ease: "easeInOut" }}
                  />
                </div>
              </div>
            )
          )}
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className={`bg-[#1a2332] rounded-xl border border-white/[0.06] p-5 ${className}`}>
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <div className="w-16 h-16 rounded-2xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center mb-4">
            <TrendingUp className="w-7 h-7 text-slate-600" />
          </div>
          <h3 className="text-sm font-medium text-slate-400 mb-1">
            No Analysis Yet
          </h3>
          <p className="text-xs text-slate-600 max-w-[200px]">
            Click on the map to select a location and run mineral prospectivity
            analysis
          </p>
        </div>
      </div>
    );
  }

  const sections = [
    { key: "spectral", label: "Spectral", content: <SpectralSection data={result.spectral} /> },
    { key: "terrain", label: "Terrain", content: <TerrainSection data={result.terrain} /> },
    { key: "proximity", label: "Proximity", content: <ProximitySection data={result.proximity} /> },
    { key: "risk", label: "Risk", content: <RiskSection data={result.risk} /> },
  ];

  return (
    <div className={`bg-[#1a2332] rounded-xl border border-white/[0.06] ${className}`}>
      {/* Header with prospectivity score */}
      <div className="p-4 border-b border-white/[0.06]">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Prospectivity Score
          </span>
          <span className="text-[10px] text-slate-600">
            {result.mineral_targets.length} targets
          </span>
        </div>
        <div className="flex items-center gap-3">
          <motion.div
            initial={{ scale: 0.5, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="relative w-14 h-14"
          >
            <svg viewBox="0 0 36 36" className="w-14 h-14 -rotate-90">
              <path
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                fill="none"
                stroke="rgba(255,255,255,0.05)"
                strokeWidth="3"
              />
              <path
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                fill="none"
                stroke={
                  result.prospectivity_score > 0.6
                    ? "#10b981"
                    : result.prospectivity_score > 0.4
                    ? "#f59e0b"
                    : "#ef4444"
                }
                strokeWidth="3"
                strokeDasharray={`${result.prospectivity_score * 100}, 100`}
                strokeLinecap="round"
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-lg font-bold font-mono text-slate-200">
                {(result.prospectivity_score * 100).toFixed(0)}
              </span>
            </div>
          </motion.div>
          <div className="flex-1 min-w-0">
            <p className="text-xs text-slate-300 leading-relaxed">
              {result.recommendation}
            </p>
          </div>
        </div>
      </div>

      {/* Collapsible sections */}
      <ScrollArea className="max-h-[500px]">
        <div className="p-4 space-y-4">
          {sections.map((section) => (
            <div key={section.key} className="border border-white/[0.04] rounded-lg overflow-hidden">
              <button
                onClick={() => toggleSection(section.key)}
                className="w-full flex items-center justify-between px-3 py-2 hover:bg-white/[0.02] transition-colors"
              >
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  {section.label}
                </span>
                {expandedSections[section.key] ? (
                  <ChevronDown className="w-3.5 h-3.5 text-slate-500" />
                ) : (
                  <ChevronRight className="w-3.5 h-3.5 text-slate-500" />
                )}
              </button>
              <AnimatePresence>
                {expandedSections[section.key] && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="px-3 pb-3">{section.content}</div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
