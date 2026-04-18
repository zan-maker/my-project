"use client";

import { MINERALS } from "@/lib/mock-data";
import type { Mineral } from "@/lib/types";
import { motion, AnimatePresence } from "framer-motion";

interface MineralSelectorProps {
  selected: Mineral[];
  onToggle: (mineral: Mineral) => void;
  className?: string;
}

const mineralKeys = Object.keys(MINERALS) as Mineral[];

export default function MineralSelector({
  selected,
  onToggle,
  className = "",
}: MineralSelectorProps) {
  return (
    <div className={`flex flex-wrap gap-2 ${className}`}>
      {mineralKeys.map((key) => {
        const info = MINERALS[key];
        const isSelected = selected.includes(key);

        return (
          <motion.button
            key={key}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => onToggle(key)}
            className={`
              relative flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium
              transition-all duration-200 cursor-pointer select-none
              border backdrop-blur-sm
              ${
                isSelected
                  ? "border-transparent shadow-lg"
                  : "border-white/10 hover:border-white/20"
              }
            `}
            style={
              isSelected
                ? {
                    background: info.bgColor,
                    borderColor: info.borderColor,
                    color: info.color,
                    boxShadow: `0 4px 20px ${info.color}25`,
                  }
                : {
                    background: "rgba(26,35,50,0.6)",
                    color: "#64748b",
                  }
            }
          >
            <span className="text-base">{info.icon}</span>
            <span>{info.name}</span>
            <AnimatePresence>
              {isSelected && (
                <motion.span
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0, opacity: 0 }}
                  className="ml-0.5 w-1.5 h-1.5 rounded-full"
                  style={{ background: info.color }}
                />
              )}
            </AnimatePresence>
          </motion.button>
        );
      })}
    </div>
  );
}
