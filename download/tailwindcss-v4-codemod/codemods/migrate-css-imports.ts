/**
 * Tailwind CSS v3 → v4: Replace @tailwind directives with @import
 *
 * Converts: @tailwind base; @tailwind components; @tailwind utilities;
 * To:       @import "tailwindcss";
 */

const codemod = (node, _ctx) => {
  const root = node.root();
  let text = root.text();

  // Check if already v4
  if (text.includes('@import "tailwindcss"')) return null;

  // Check if any @tailwind directives exist
  if (!text.includes("@tailwind")) return null;

  // Remove all @tailwind directives
  text = text.replace(/@tailwind\s+\w+;\s*/g, "");

  // Add @import "tailwindcss" at the top
  text = '@import "tailwindcss";\n' + text;

  return root.commitEdits([root.replace(text)]);
};

export default codemod;
