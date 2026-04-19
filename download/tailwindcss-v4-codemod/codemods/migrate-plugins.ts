/**
 * Tailwind CSS v3 → v4: Migrate plugins to CSS imports
 */

const PLUGIN_MAP = {
  "tailwindcss-animate": '// ADD to main CSS: @import "tw-animate-css";',
  "@tailwindcss/typography": '// ADD to main CSS: @import "@tailwindcss/typography";',
  "@tailwindcss/forms": '// ADD to main CSS: @import "@tailwindcss/forms";',
  "@tailwindcss/aspect-ratio": '// ADD to main CSS: @import "@tailwindcss/aspect-ratio";',
  "@tailwindcss/container-queries": '// ADD to main CSS: @import "@tailwindcss/container-queries";',
};

const BUILTIN_PLUGINS = ["@tailwindcss/line-clamp", "tailwindcss-line-clamp"];

const codemod = (node, _ctx) => {
  const filename = node.filename();
  const isConfig =
    filename.endsWith("tailwind.config.js") ||
    filename.endsWith("tailwind.config.ts") ||
    filename.endsWith("tailwind.config.mjs") ||
    filename.endsWith("tailwind.config.cjs");

  if (!isConfig) return null;

  const root = node.root();
  let text = root.text();

  const pluginsSection = text.match(/plugins\s*:\s*\[([\s\S]*?)\]/);
  if (!pluginsSection) return null;

  const pluginBody = pluginsSection[1];
  const instructions = [];

  for (const [plugin, note] of Object.entries(PLUGIN_MAP)) {
    if (pluginBody.includes(plugin)) {
      instructions.push(note);
    }
  }

  for (const plugin of BUILTIN_PLUGINS) {
    if (pluginBody.includes(plugin)) {
      instructions.push("// REMOVE: " + plugin + " — built into Tailwind v4");
    }
  }

  // Generic plugin detection
  const requireMatches = pluginBody.matchAll(/require\s*\(\s*['"]([^'"]+)['"]\s*\)/g);
  for (const [, plugin] of requireMatches) {
    if (!PLUGIN_MAP[plugin] && !BUILTIN_PLUGINS.includes(plugin)) {
      instructions.push('// CHECK: "' + plugin + '" — may need manual v4 migration');
    }
  }

  if (instructions.length === 0) return null;

  const note = "\n// ── Tailwind v4 Plugin Migration ──\n" +
    instructions.join("\n") + "\n";

  text = text + note;
  return root.commitEdits([root.replace(text)]);
};

export default codemod;
