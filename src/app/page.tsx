"use client";

import { useState, useCallback, useRef } from "react";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import {
  Microscope,
  Layers,
  Zap,
  Globe2,
  Cpu,
  RotateCcw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const InteractiveMap = dynamic(
  () => import("@/components/dashboard/InteractiveMap"),
  { ssr: false, loading: () => <div className="absolute inset-0 bg-[#1a2332] rounded-xl animate-pulse" /> }
);
import MineralSelector from "@/components/dashboard/MineralSelector";
import AnalysisPanel from "@/components/dashboard/AnalysisPanel";
import ChatInterface from "@/components/dashboard/ChatInterface";
import ToolCallLog from "@/components/dashboard/ToolCallLog";
import ReportPanel from "@/components/dashboard/ReportPanel";
import {
  generateMockAnalysis,
  generateMockChatResponse,
  generateMockToolCallChain,
  generateMockReport,
} from "@/lib/mock-data";
import type {
  Mineral,
  Location,
  AnalysisResult,
  ChatMessage,
  ToolCall,
  ProspectivityReport,
} from "@/lib/types";

// Generate a unique ID
function uid() {
  return Math.random().toString(36).slice(2, 10);
}

// All minerals
const ALL_MINERALS: Mineral[] = ["lithium", "cobalt", "rare_earth", "copper", "nickel"];

export default function DashboardPage() {
  // State
  const [selectedLocation, setSelectedLocation] = useState<Location | null>(null);
  const [selectedMinerals, setSelectedMinerals] = useState<Mineral[]>([...ALL_MINERALS]);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [toolCalls, setToolCalls] = useState<ToolCall[]>([]);
  const [report, setReport] = useState<ProspectivityReport | null>(null);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [isFullscreen] = useState(false);
  const [rightTab, setRightTab] = useState("analysis");

  const analysisTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Run analysis
  const runAnalysis = useCallback(
    async (lat: number, lon: number, minerals: Mineral[]) => {
      if (isAnalyzing) return;

      setIsAnalyzing(true);
      setAnalysisResult(null);
      setReport(null);

      // Simulate tool call chain appearing sequentially
      const chain = generateMockToolCallChain(lat, lon, minerals);
      setToolCalls([]);

      // Stagger tool call appearances
      chain.forEach((tc, i) => {
        setTimeout(() => {
          setToolCalls((prev) => {
            const updated = [...prev];
            const idx = updated.findIndex((t) => t.id === tc.id);
            if (idx >= 0) {
              updated[idx] = { ...tc, status: "completed" };
            } else {
              updated.push({ ...tc, status: "running" });
            }
            return updated;
          });

          // Mark as completed after a delay
          setTimeout(() => {
            setToolCalls((prev) =>
              prev.map((t) =>
                t.id === tc.id ? { ...t, status: "completed" as const } : t
              )
            );
          }, 400 + Math.random() * 800);
        }, i * 600);
      });

      // Simulate API call delay
      analysisTimeoutRef.current = setTimeout(() => {
        const result = generateMockAnalysis(lat, lon, minerals);
        setAnalysisResult(result);
        setIsAnalyzing(false);
        setRightTab("analysis");
      }, chain.length * 600 + 1200);
    },
    [isAnalyzing]
  );

  // Handle mineral toggle
  const handleMineralToggle = useCallback(
    (mineral: Mineral) => {
      setSelectedMinerals((prev) => {
        const next = prev.includes(mineral)
          ? prev.filter((m) => m !== mineral)
          : [...prev, mineral];
        if (next.length === 0) return prev;
        return next;
      });
    },
    []
  );

  // Handle chat message
  const handleSendMessage = useCallback(
    async (message: string) => {
      const userMsg: ChatMessage = {
        id: uid(),
        role: "user",
        content: message,
        timestamp: Date.now(),
      };

      setChatMessages((prev) => [...prev, userMsg]);
      setIsTyping(true);

      // Simulate response delay
      setTimeout(() => {
        const response = generateMockChatResponse(
          message,
          selectedLocation?.lat,
          selectedLocation?.lon
        );

        const assistantMsg: ChatMessage = {
          id: uid(),
          role: "assistant",
          content: response,
          tool_calls:
            selectedLocation && Math.random() > 0.5
              ? generateMockToolCallChain(
                  selectedLocation.lat,
                  selectedLocation.lon,
                  selectedMinerals
                ).slice(0, 2 + Math.floor(Math.random() * 3))
              : [],
          timestamp: Date.now(),
        };

        setChatMessages((prev) => [...prev, assistantMsg]);
        setIsTyping(false);
      }, 1200 + Math.random() * 1500);
    },
    [selectedLocation, selectedMinerals]
  );

  // Generate report
  const handleGenerateReport = useCallback(() => {
    if (!analysisResult) return;

    setIsGeneratingReport(true);

    setTimeout(() => {
      const reportData = generateMockReport(
        analysisResult.location.lat,
        analysisResult.location.lon,
        analysisResult.mineral_targets,
        analysisResult
      );
      setReport(reportData);
      setIsGeneratingReport(false);
      setRightTab("report");
    }, 5000);
  }, [analysisResult]);

  // Reset
  const handleReset = useCallback(() => {
    if (analysisTimeoutRef.current) clearTimeout(analysisTimeoutRef.current);
    setSelectedLocation(null);
    setAnalysisResult(null);
    setChatMessages([]);
    setToolCalls([]);
    setReport(null);
    setIsAnalyzing(false);
    setIsTyping(false);
    setIsGeneratingReport(false);
  }, []);

  return (
    <div className="min-h-screen flex flex-col" style={{ background: "#0a0f1a" }}>
      {/* ===== HEADER ===== */}
      <header className="sticky top-0 z-50 border-b border-white/[0.06] backdrop-blur-xl" style={{ background: "rgba(10,15,26,0.85)" }}>
        <div className="max-w-[1920px] mx-auto px-4 lg:px-6 h-14 flex items-center justify-between">
          {/* Logo & Title */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <Microscope className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-bold text-slate-100 tracking-tight">
                MineLens AI
              </h1>
              <p className="text-[10px] text-slate-500 -mt-0.5">
                Gemma 4 Good Hackathon — Live Demo
              </p>
            </div>
          </div>

          {/* Center: Mineral selector (desktop) */}
          <div className="hidden md:flex items-center gap-2">
            <MineralSelector
              selected={selectedMinerals}
              onToggle={handleMineralToggle}
            />
          </div>

          {/* Right: Actions */}
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleReset}
              className="text-slate-400 hover:text-slate-200 h-8 text-xs"
            >
              <RotateCcw className="w-3.5 h-3.5 mr-1.5" />
              Reset
            </Button>
            <Button
              size="sm"
              onClick={handleGenerateReport}
              disabled={!analysisResult || isGeneratingReport}
              className="bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 border border-emerald-500/20 h-8 text-xs"
            >
              <Layers className="w-3.5 h-3.5 mr-1.5" />
              Generate Report
            </Button>
            <Badge
              variant="outline"
              className="hidden lg:flex text-[9px] border-emerald-500/30 text-emerald-400 bg-emerald-500/5 h-6"
            >
              <Cpu className="w-2.5 h-2.5 mr-1" />
              Gemma 4 Function Calling
            </Badge>
          </div>
        </div>
      </header>

      {/* ===== MAIN CONTENT ===== */}
      <main className="flex-1 max-w-[1920px] mx-auto w-full p-4 lg:p-6">
        {/* Mobile mineral selector */}
        <div className="md:hidden mb-4">
          <MineralSelector
            selected={selectedMinerals}
            onToggle={handleMineralToggle}
          />
        </div>

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 lg:gap-5 h-[calc(100vh-7rem)]">
          {/* LEFT COLUMN: Map + Tool Log */}
          <div className="lg:col-span-7 xl:col-span-8 flex flex-col gap-4 lg:gap-5 min-h-0">
            {/* Map */}
            <div className="relative flex-1 min-h-[300px] lg:min-h-0">
              <InteractiveMap
                selectedLocation={selectedLocation}
                onLocationSelect={(lat, lon) =>
                  runAnalysis(lat, lon, selectedMinerals)
                }
                className="absolute inset-0"
              />
            </div>

            {/* Tool Call Log */}
            <div className="h-[220px] lg:h-[240px] flex-shrink-0">
              <ToolCallLog toolCalls={toolCalls} className="h-full" />
            </div>
          </div>

          {/* RIGHT COLUMN: Analysis + Chat + Report */}
          <div className="lg:col-span-5 xl:col-span-4 flex flex-col gap-4 lg:gap-5 min-h-0">
            {/* Tabbed Panel */}
            <Tabs
              value={rightTab}
              onValueChange={setRightTab}
              className="flex flex-col flex-1 min-h-0"
            >
              <TabsList className="bg-[#1a2332]/80 border border-white/[0.06] w-full rounded-xl p-1 h-10">
                <TabsTrigger
                  value="analysis"
                  className="text-xs data-[state=active]:bg-white/[0.06] data-[state=active]:text-emerald-400 rounded-lg h-8 flex-1 gap-1.5"
                >
                  <Zap className="w-3 h-3" />
                  Analysis
                </TabsTrigger>
                <TabsTrigger
                  value="chat"
                  className="text-xs data-[state=active]:bg-white/[0.06] data-[state=active]:text-emerald-400 rounded-lg h-8 flex-1 gap-1.5"
                >
                  <Globe2 className="w-3 h-3" />
                  AI Geologist
                  {chatMessages.length > 0 && (
                    <Badge className="w-4 h-4 p-0 text-[8px] bg-emerald-500/20 text-emerald-400 border-0">
                      {chatMessages.length}
                    </Badge>
                  )}
                </TabsTrigger>
                <TabsTrigger
                  value="report"
                  className="text-xs data-[state=active]:bg-white/[0.06] data-[state=active]:text-emerald-400 rounded-lg h-8 flex-1 gap-1.5"
                >
                  <Layers className="w-3 h-3" />
                  Report
                  {report && (
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                  )}
                </TabsTrigger>
              </TabsList>

              <TabsContent value="analysis" className="flex-1 mt-3 min-h-0">
                <AnalysisPanel
                  result={analysisResult}
                  isLoading={isAnalyzing}
                  className="h-full"
                />
              </TabsContent>

              <TabsContent value="chat" className="flex-1 mt-3 min-h-0">
                <ChatInterface
                  messages={chatMessages}
                  isTyping={isTyping}
                  onSendMessage={handleSendMessage}
                  className="h-full"
                />
              </TabsContent>

              <TabsContent value="report" className="flex-1 mt-3 min-h-0">
                <ReportPanel
                  report={report}
                  isGenerating={isGeneratingReport}
                  onGenerate={handleGenerateReport}
                  className="h-full"
                />
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </main>

      {/* ===== FOOTER ===== */}
      <footer className="border-t border-white/[0.04] py-3 px-4">
        <div className="max-w-[1920px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <div className="flex items-center gap-3 text-[10px] text-slate-600">
            <span className="flex items-center gap-1">
              <Microscope className="w-3 h-3" />
              MineLens AI
            </span>
            <span>•</span>
            <span>Gemma 4 Good Hackathon</span>
            <span>•</span>
            <span>Powered by Google Gemma 4 Function Calling</span>
          </div>
          <div className="flex items-center gap-3 text-[10px] text-slate-600">
            <span>Data: USGS MRDS, ASTER, Sentinel-2</span>
            <span>•</span>
            <span className="text-emerald-500/60">Demo Mode</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
