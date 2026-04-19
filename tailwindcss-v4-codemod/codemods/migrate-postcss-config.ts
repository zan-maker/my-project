/**
 * Tailwind CSS v3 → v4: Update PostCSS configuration
 *
 * Replaces postcss.config.js content with @tailwindcss/postcss.
 */

const codemod = (node, _ctx) => {
  const filename = node.filename().toLowerCase();
  const isPostcss =
    filename.endsWith("postcss.config.js") ||
    filename.endsWith("postcss.config.mjs") ||
    filename.endsWith("postcss.config.cjs");

  if (!isPostcss) return null;

  const text = node.source();
  if (text.includes("@tailwindcss/postcss")) return null;

  const isEsm = text.includes("export default") || filename.endsWith(".mjs");
  const newContent = isEsm
    ? 'const config = {\n  plugins: ["@tailwindcss/postcss"],\n}\n\nexport default config\n'
    : 'module.exports = {\n  plugins: ["@tailwindcss/postcss"],\n}\n';

  const root = node.root();
  return root.commitEdits([root.replace(newContent)]);
};

export default codemod;
