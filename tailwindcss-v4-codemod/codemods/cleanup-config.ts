/**
 * Tailwind CSS v3 → v4: Remove deprecated config options
 *
 * Removes: content, purge, mode, separator from tailwind.config
 */

const codemod = (node, _ctx) => {
  const filename = node.filename();
  const isConfig =
    filename.endsWith("tailwind.config.js") ||
    filename.endsWith("tailwind.config.ts") ||
    filename.endsWith("tailwind.config.mjs") ||
    filename.endsWith("tailwind.config.cjs");

  if (!isConfig) return null;

  const root = node.root();
  const text = root.text();

  const lines = text.split("\n");
  const newLines = [];
  let changed = false;

  for (const line of lines) {
    const trimmed = line.trim();

    // Remove content: [...] lines
    if (trimmed.startsWith("content:") && trimmed.includes("[")) {
      newLines.push("  // REMOVED: content — v4 auto-detects source files");
      changed = true;
    }
    // Remove purge: [...] lines
    else if (trimmed.startsWith("purge:") && trimmed.includes("[")) {
      newLines.push("  // REMOVED: purge — renamed to content in v3, removed in v4");
      changed = true;
    }
    // Comment out mode
    else if (trimmed.startsWith("mode:") && (trimmed.includes('"') || trimmed.includes("'"))) {
      newLines.push("  // REMOVED: mode — JIT is always-on in v4");
      changed = true;
    }
    // Comment out separator
    else if (trimmed.startsWith("separator:") && (trimmed.includes('"') || trimmed.includes("'"))) {
      newLines.push("  // REMOVED: separator — not supported in v4");
      changed = true;
    }
    // Comment out safelist
    else if (trimmed.startsWith("safelist:") && trimmed.includes("[")) {
      newLines.push("  // REMOVED: safelist — use @source inline in CSS");
      changed = true;
    }
    else {
      newLines.push(line);
    }
  }

  return changed ? root.commitEdits([root.replace(newLines.join("\n"))]) : null;
};

export default codemod;
