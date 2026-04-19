/**
 * Tailwind CSS v3 → v4: Rename removed/deprecated utilities
 *
 * shadow-sm → shadow-xs, bare shadow → shadow-sm
 * blur-sm → blur-xs, bare blur → blur-sm
 * focus:outline-none → focus:outline-hidden
 * ring (bare) → ring-3
 * bg-opacity-*, text-opacity-*, etc. → TW4_REVIEW_ markers
 *
 * Uses text-based replacement for broad language support.
 */

const codemod = (node, _ctx) => {
  const root = node.root();
  let text = root.text();
  let modified = false;

  // Shadow: shadow-sm → shadow-xs first, then bare shadow → shadow-sm
  if (text.includes("shadow-sm")) {
    text = text.replaceAll("shadow-sm", "shadow-xs");
    modified = true;
  }
  if (/shadow/.test(text)) {
    text = text.replaceAll(/\bshadow\b(?![-/\d])/g, "shadow-sm");
    modified = true;
  }

  // Blur: blur-sm → blur-xs first, then bare blur → blur-sm
  if (text.includes("blur-sm")) {
    text = text.replaceAll("blur-sm", "blur-xs");
    modified = true;
  }
  if (/blur(?!-)/.test(text)) {
    text = text.replaceAll(/\bblur\b(?![-/\d])/g, "blur-sm");
    modified = true;
  }

  // Outline
  if (text.includes("outline-none")) {
    text = text.replaceAll("outline-none", "outline-hidden");
    modified = true;
  }

  // Ring: bare ring → ring-3
  if (/ring/.test(text)) {
    text = text.replaceAll(/\bring\b(?![-/\d])/g, "ring-3");
    modified = true;
  }

  // Opacity utilities: flag for manual review
  const opacityPrefixes = [
    "bg-opacity-", "text-opacity-", "border-opacity-",
    "divide-opacity-", "ring-opacity-", "placeholder-opacity-",
  ];
  for (const prefix of opacityPrefixes) {
    if (text.includes(prefix)) {
      text = text.replaceAll(prefix, prefix + "TW4_REVIEW_");
      modified = true;
    }
  }

  return modified ? root.commitEdits([root.replace(text)]) : null;
};

export default codemod;
