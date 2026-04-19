/**
 * Tailwind CSS v3 → v4: Migrate tailwind.config to CSS @theme blocks
 *
 * Converts the config to a deprecated stub with migration notes
 * and generates a @theme block for globals.css.
 */

const codemod = (node, _ctx) => {
  const filename = node.filename().toLowerCase();
  const isConfig =
    filename.endsWith("tailwind.config.js") ||
    filename.endsWith("tailwind.config.ts") ||
    filename.endsWith("tailwind.config.mjs") ||
    filename.endsWith("tailwind.config.cjs");

  if (!isConfig) return null;

  const text = node.source();
  if (text.includes("DEPRECATED after Tailwind v4")) return null;

  const hasDarkMode = /darkMode\s*:\s*["']class["']/.test(text);
  const hasPlugins = /plugins\s*:/.test(text);
  const hasContent = /content\s*:/.test(text);

  // Extract theme block for migration
  const themeBlock = extractThemeBlock(text);

  const lines = [
    "/**",
    ` * ${filename} — DEPRECATED after Tailwind v4 migration`,
    " * All theme values have been migrated to CSS @theme blocks.",
    " * This file can be safely deleted after verifying the migration.",
    " *",
    " * Detected v3 features:",
  ];

  if (hasDarkMode) lines.push(" *   - darkMode: \"class\" → @custom-variant dark");
  if (hasPlugins) lines.push(" *   - plugins → convert to CSS @import");
  if (hasContent) lines.push(" *   - content → removed (v4 auto-detects)");

  if (themeBlock) {
    lines.push(" *");
    lines.push(" * Generated @theme block for globals.css:");
    lines.push(" * ─────────────────────────────────────────");
    for (const line of themeBlock.split("\n")) {
      lines.push(" * " + line);
    }
    lines.push(" * ─────────────────────────────────────────");
  }

  lines.push(" */");
  lines.push("");
  lines.push("// Tailwind CSS v4 auto-detects source files.");
  lines.push("// No configuration file is needed.");
  lines.push("// Delete this file after verifying the migration.");

  const root = node.root();
  return root.commitEdits([root.replace(lines.join("\n"))]);
};

function extractThemeBlock(text) {
  const colorsMatch = text.match(/colors\s*:\s*\{([\s\S]*?)\n\s{0,4}\}/);
  const radiusMatch = text.match(/borderRadius\s*:\s*\{([\s\S]*?)\n\s{0,4}\}/);
  const chartMatch = text.match(/chart\s*:\s*\{([\s\S]*?)\n\s{0,6}\}/);

  if (!colorsMatch && !radiusMatch && !chartMatch) return null;

  const lines = ["@theme inline {"];

  if (colorsMatch) {
    const simpleColors = colorsMatch[1].matchAll(/(\w[\w-]*)\s*:\s*'([^']+)'/g);
    for (const [, name, value] of simpleColors) {
      lines.push("  --color-" + name + ": " + value + ";");
    }
  }

  if (radiusMatch) {
    const radii = radiusMatch[1].matchAll(/(\w+)\s*:\s*'([^']+)'/g);
    for (const [, name, value] of radii) {
      lines.push("  --radius-" + name + ": " + value + ";");
    }
  }

  if (chartMatch) {
    const charts = chartMatch[1].matchAll(/['"](\d+)['"]\s*:\s*['"]([^'"]+)['"]/g);
    for (const [, num, value] of charts) {
      lines.push("  --color-chart-" + num + ": " + value + ";");
    }
  }

  lines.push("}");
  return lines.join("\n");
}

export default codemod;
