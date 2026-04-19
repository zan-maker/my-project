# tailwindcss-v3-to-v4

> Automated codemod to migrate Tailwind CSS projects from v3 to v4.

[![Codemod Registry](https://img.shields.io/badge/codemod-registry-green)](https://codemod.com/registry)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

This codemod automates the migration from Tailwind CSS v3 to v4, handling the most common breaking changes. It's designed to cover **80%+ of the migration deterministically**, with clear instructions for the remaining edge cases that require manual review.

## What It Does

### Phase 1: Config Migration
Converts `tailwind.config.js/ts` theme extensions to CSS `@theme` blocks:
- `theme.extend.colors` → `@theme inline { --color-*: ... }`
- `theme.extend.borderRadius` → `@theme inline { --radius-*: ... }`
- `theme.extend.chart` → `@theme inline { --color-chart-*: ... }`

### Phase 2: PostCSS Config
Replaces the PostCSS configuration:
- Removes `tailwindcss` and `autoprefixer` plugins
- Adds `@tailwindcss/postcss` as the sole plugin

### Phase 3: CSS Import Migration
Updates CSS directives:
- `@tailwind base; @tailwind components; @tailwind utilities;` → `@import "tailwindcss";`

### Phase 4: Utility Renaming
Renames deprecated/changed utilities:
- `shadow-sm` → `shadow-xs`, bare `shadow` → `shadow-sm`
- `blur-sm` → `blur-xs`, bare `blur` → `blur-sm`
- `focus:outline-none` → `focus:outline-hidden`
- Bare `ring` → `ring-3` (v4 defaults ring width to 1px)
- Opacity utilities (`bg-opacity-*`, etc.) → flagged for manual review

### Phase 5: Gradient Migration
- `bg-gradient-to-r` → `bg-linear-to-r`
- `via-transparent` → removed (v4 preserves gradient values)
- All 8 gradient directions handled

### Phase 6: Dark Mode
- Adds `@custom-variant dark (&:is(.dark *));` to CSS

### Phase 7: Plugin Migration
- `tailwindcss-animate` → `@import "tw-animate-css"`
- `@tailwindcss/line-clamp` → removed (built into v4)
- Other plugins get migration instructions

### Phase 8: Config Cleanup
- Removes `content`, `mode`, `separator`, `safelist`, `blocklist`, `purge`
- Comments out `prefix` and `important` with v4 alternatives

## Usage

### Option 1: From Codemod Registry
```bash
npx codemod tailwindcss-v3-to-v4
```

### Option 2: Local Clone
```bash
git clone https://github.com/zan-maker/tailwindcss-v4-codemod.git
cd tailwindcss-v4-codemod
npx codemod workflow run -w codemods/workflow.yaml -t /path/to/your/project
```

### Option 3: Quick Test
```bash
# Test a single transform on one file
npx codemod jssg run -p codemods/migrate-gradients.ts -t ./src/app/page.tsx
```

## Post-Migration Checklist

After running the codemod, complete these manual steps:

1. **Update dependencies:**
   ```bash
   npm install tailwindcss@4 @tailwindcss/postcss
   npm uninstall autoprefixer  # v4 handles prefixing automatically
   ```

2. **Review `globals.css`:**
   - Verify `@theme inline { ... }` contains all your custom theme values
   - Check `@custom-variant dark` is present if you use dark mode

3. **Fix opacity utilities** (flagged as `-REVIEW_ME`):
   - `bg-opacity-50` → use `bg-black/50` or the specific color with opacity
   - `text-opacity-75` → use `text-white/75` or `text-current/75`

4. **Verify ring widths:**
   - The codemod replaces bare `ring` with `ring-3` to preserve v3 appearance
   - Review each instance to ensure it looks correct

5. **Check border colors:**
   - v4 defaults `border` to `currentColor` instead of themed color
   - Add `border-border` (shadcn) or explicit color if borders look wrong

6. **Test your app:**
   ```bash
   npm run dev
   ```
   - Check all pages, especially those with dark mode, gradients, and shadows

7. **Delete `tailwind.config.ts`** (optional):
   - Once verified, the config file can be safely removed

## Compatibility

| Framework | Status |
|-----------|--------|
| Next.js (App Router) | ✅ Full support |
| Next.js (Pages Router) | ✅ Full support |
| Vite + React | ✅ Full support |
| Vite + Vue | ✅ Full support |
| Remix | ✅ Full support |
| shadcn/ui projects | ✅ Optimized for |

## Known Limitations

- **Dynamic class construction** (`\`bg-${color}-500\``) cannot be detected statically
- **Conditional opacity utilities** require manual pairing with the correct color
- **Custom plugin implementations** (not from Tailwind ecosystem) need manual migration
- **CSS-in-JS libraries** using Tailwind classes may need additional transforms

## Tech Stack

- [Codemod CLI](https://docs.codemod.com/cli) — orchestration and publishing
- [jssg](https://docs.codemod.com/cli#jssg) — TypeScript AST transforms
- [ast-grep](https://ast-grep.github.io/) — pattern-based AST matching

## Related Resources

- [Tailwind CSS v4 Upgrade Guide](https://tailwindcss.com/docs/upgrade-guide)
- [Tailwind CSS v4 Blog Post](https://tailwindcss.com/blog/tailwindcss-v4)
- [@tailwindcss/upgrade](https://www.npmjs.com/package/@tailwindcss/upgrade) — Official upgrade tool
- [Codemod Documentation](https://docs.codemod.com/)

## License

MIT
