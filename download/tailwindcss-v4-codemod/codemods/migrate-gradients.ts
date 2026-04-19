/**
 * Tailwind CSS v3 → v4: Migrate gradient utilities
 *
 * bg-gradient-to-r → bg-linear-to-r (all 8 directions)
 * via-transparent → remove
 *
 * Uses text-based replacement since className strings are JSX attributes.
 */

const GRADIENT_MAP = {
  "bg-gradient-to-r": "bg-linear-to-r",
  "bg-gradient-to-l": "bg-linear-to-l",
  "bg-gradient-to-t": "bg-linear-to-t",
  "bg-gradient-to-b": "bg-linear-to-b",
  "bg-gradient-to-tr": "bg-linear-to-tr",
  "bg-gradient-to-tl": "bg-linear-to-tl",
  "bg-gradient-to-br": "bg-linear-to-br",
  "bg-gradient-to-bl": "bg-linear-to-bl",
};

const codemod = (node, _ctx) => {
  const root = node.root();
  let text = root.text();

  let modified = false;

  for (const [old, new_] of Object.entries(GRADIENT_MAP)) {
    if (text.includes(old)) {
      text = text.replaceAll(old, new_);
      modified = true;
    }
  }

  // Remove via-transparent
  if (text.includes("via-transparent")) {
    text = text.replace(/\s+via-transparent/g, "");
    modified = true;
  }

  return modified ? root.commitEdits([root.replace(text)]) : null;
};

export default codemod;
