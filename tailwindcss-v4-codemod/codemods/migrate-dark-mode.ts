/**
 * Tailwind CSS v3 → v4: Migrate dark mode configuration
 *
 * Adds @custom-variant dark (&:is(.dark *)); to main CSS file.
 */

const codemod = (node, _ctx) => {
  const filename = node.filename().toLowerCase();
  const isMainCss =
    (filename.includes("global") || filename.includes("index") ||
     filename.includes("main") || filename.includes("app")) &&
    filename.endsWith(".css");

  if (!isMainCss) return null;

  const root = node.root();
  let text = root.text();

  if (text.includes("@custom-variant dark")) return null;
  if (!text.includes("dark:")) return null;

  // Insert @custom-variant after @import "tailwindcss"
  if (text.includes('@import "tailwindcss"')) {
    text = text.replace(
      /(@import\s+"tailwindcss";)/,
      '$1\n\n@custom-variant dark (&:is(.dark *));'
    );
  } else {
    text = '@custom-variant dark (&:is(.dark *));\n\n' + text;
  }

  return root.commitEdits([root.replace(text)]);
};

export default codemod;
