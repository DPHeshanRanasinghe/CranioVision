# CranioVision — Design Upgrade Brief

The CranioVision frontend now works end-to-end. This document describes the
visual design upgrade that brings the UI to production-grade quality.

Read this BEFORE making changes. The functional code is working — do not
break it. Style only.

## Scope Of This Pass

**In scope:**
- Visual polish of every component
- Typography upgrade
- Color palette refinement
- Card design, shadows, borders, spacing
- Hover states, transitions, animations
- Layout proportions and visual hierarchy
- Merging useful styling patterns from `frontend-v0/`
- After successful merge, DELETE `frontend-v0/`

**Out of scope (DO NOT TOUCH):**
- `lib/api.ts` — API client logic
- `lib/types.ts` — TypeScript schemas
- `Brain3DViewer.tsx` plotly logic — visualization is working
- `mesh_extractor.py` and backend code — pipeline is solid
- The job state, SSE/polling fallback, error handling
- The two-mode viewer architecture (segmentation mode / XAI mode)

If a styling change requires altering API or state logic, STOP and ask first.

## Visual Direction

The aesthetic should land between **Linear.app** (clean SaaS minimalism) and
**Stripe Dashboard** (data-dense but elegant) and **Apple HealthKit** (medical
seriousness without being sterile). Think:

- Confident but not flashy
- Information-rich but not cluttered
- Medical/clinical but not boring
- Premium feel — like a tool that costs money

## Color System

Replace the current ad-hoc colors with a coherent system. Use these or close
equivalents:

```css
/* Surfaces */
--surface-page: #FAFAFA           /* Page background — warmer than pure white */
--surface-card: #FFFFFF
--surface-card-hover: #F8F9FB
--surface-elevated: #FFFFFF       /* For modals, overlays */
--surface-muted: #F1F3F5

/* Borders */
--border-subtle: #E5E7EB
--border-default: #D1D5DB
--border-strong: #9CA3AF

/* Text */
--text-primary: #111827
--text-secondary: #4B5563
--text-tertiary: #6B7280
--text-muted: #9CA3AF

/* Brand — clinical navy/blue family */
--brand-50: #EFF6FF
--brand-100: #DBEAFE
--brand-500: #3B82F6
--brand-600: #2563EB
--brand-700: #1D4ED8
--brand-900: #1E3A8A
--brand-navy: #003366            /* The deepest signature color */

/* Semantic — risk colors (already defined for eloquent cortex) */
--risk-high: #DC2626              /* Red */
--risk-moderate: #F59E0B          /* Amber */
--risk-low: #FBBF24               /* Yellow */
--risk-minimal: #10B981           /* Green */

/* Tumor classes */
--tumor-edema: #FBBF24            /* Yellow */
--tumor-enhancing: #DC2626        /* Red */
--tumor-necrotic: #2563EB         /* Blue */

/* Shadow tokens */
--shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.04)
--shadow-md: 0 4px 6px -1px rgba(15, 23, 42, 0.06), 0 2px 4px -2px rgba(15, 23, 42, 0.04)
--shadow-lg: 0 10px 15px -3px rgba(15, 23, 42, 0.08), 0 4px 6px -4px rgba(15, 23, 42, 0.04)
--shadow-xl: 0 20px 25px -5px rgba(15, 23, 42, 0.1), 0 8px 10px -6px rgba(15, 23, 42, 0.04)
```

Apply via Tailwind extended theme in `tailwind.config.js`, not inline.

## Typography

Replace any default sans-serif with **Inter** (already a free Google Font).
Optionally use **Geist** if v0-v0 used it — that's a reasonable choice too.

```typescript
// In layout.tsx
import { Inter } from 'next/font/google';
const inter = Inter({ subsets: ['latin'] });
```

Type scale:
- **Display** — `text-4xl font-bold tracking-tight` (36px / 700) — for the
  big agreement banner stat
- **H1** — `text-2xl font-semibold tracking-tight` (24px / 600) — page title
- **H2** — `text-lg font-semibold tracking-tight` (18px / 600) — card titles
- **H3** — `text-sm font-semibold` (14px / 600) — subheadings
- **Body** — `text-sm` (14px / 400) — most text
- **Caption** — `text-xs text-text-tertiary` (12px / 400) — metadata
- **Numerals** — always use `tabular-nums font-mono` for percentages, volumes,
  Dice scores, distances. `font-mono` Inter has poor tabular figures — use
  JetBrains Mono or Geist Mono for numbers specifically

## Layout Refinements

**Page-level:**
- Max width 1280px (current `max-w-7xl` is fine), centered
- Generous side padding: 24px on mobile, 48px on desktop
- Vertical rhythm: 32px between major sections, 24px between cards

**Header:**
- Sticky at top
- 64-72px tall
- Subtle border-bottom + slight shadow on scroll
- Brain icon + "CranioVision" wordmark
- Small links right-aligned (Documentation, GitHub)

**Cards:**
- Border radius 12px (slightly more rounded than default)
- 1px border in `border-subtle` color
- White background
- Padding: 24px
- Subtle shadow on default state, deeper shadow on hover
- Smooth transition (200ms ease)

**Section spacing:**
```
Header (sticky)
[32px gap]
Upload + Progress row (grid 2 col, 24px gap)
[32px gap]
Model picker bar (full width)
[24px gap]
3D viewer + Metrics (grid 8/4 col, 24px gap)
[32px gap]
Anatomy + Eloquent (grid 6/6 col, 24px gap)
[32px gap]
Agreement banner (full width)
Footer
```

## Component-Specific Polish

### UploadCard
- Drop zone: dashed border 2px in `brand-200`, transitions to solid `brand-500`
  with `bg-brand-50` while dragging
- Upload icon: 48px, `brand-500`, subtle bounce animation while dragging
- "or click to browse files" — make it a more visible button-like text link
- Demo dropdown: full-width select, custom-styled (not native select)
- Selected demo case: small "Run demo" button to the right of the dropdown

### ProgressCard (the right side of top row)
- Idle state: gradient placeholder with subtle pulse animation
- Active state:
  - Big progress bar with smooth gradient (brand-500 → brand-700)
  - Stage list with check marks — animate the check on transition
  - Use `framer-motion` for stage transitions if not already installed
- Completed state: subtle green tick + "Analysis complete in Xs"
- Error state: clean red banner, NOT alarming red — just informative

### ModelPicker (4 tabs)
- Pill-style buttons in a horizontal row
- Selected: `bg-brand-600 text-white shadow-md`
- Unselected: `bg-white border border-subtle hover:border-brand-300`
- Each pill shows: model name (bold), Dice score (mono small), volume (caption)
- Smooth color transition on hover/select (200ms)

### ViewerPanel (3D viewer container)
- The viewer canvas should feel "premium":
  - Border 1px in `border-subtle`
  - Inner canvas: `bg-gradient-to-br from-slate-900 to-slate-800` — NOT
    pure black, slight gradient for depth
  - Subtle inner glow / vignette around the brain
- Toolbar above viewer:
  - Mode badge ("Segmentation Mode" / "XAI Mode") in top-left
  - Toggle buttons in top-right
  - Use ghost buttons (transparent until hover) for less visual noise
- Opacity sliders: hide in a collapsible "View options" panel below the viewer,
  not always visible
- Caption below viewer: smaller, `text-text-tertiary`

### MetricsPanel
- Volume numbers: BIG and `tabular-nums font-mono`, with units smaller next to them
  ```
  145.46
  cm³
  ```
- Use a subtle horizontal divider between volume and Dice sections
- Dice scores in a clean 2-column key-value grid

### AnatomyCard
- Pie chart: smaller, take less vertical space
- Top regions: use a subtle bar visualization (mini horizontal bars) instead
  of just numbers — more scannable
- Bars colored by `brand-500` with varying opacity by percentage

### EloquentCard
- Each row: clean horizontal layout
  - Region name (left)
  - Distance (center, monospace)
  - Risk badge (right)
- Risk badges: pill-shaped, semi-transparent backgrounds:
  - HIGH: `bg-red-100 text-red-800 border border-red-200`
  - MODERATE: `bg-amber-100 text-amber-800 border border-amber-200`
  - LOW: `bg-yellow-100 text-yellow-800 border border-yellow-200`
  - MINIMAL: `bg-green-100 text-green-800 border border-green-200`
- Clinical alert at bottom: amber background, AlertCircle icon, distinctive
  but not alarming

### AgreementBanner
- This is the visual climax of the page. Make it count.
- Two-column layout (current) but with stronger visual weight:
  - Left column: the 98.4% number is HUGE — `text-7xl font-bold` —
    in `brand-navy`. Subtitle smaller below.
  - Right column: explanatory text, then the action buttons
- Subtle vertical divider between columns (1px in `border-subtle`)
- Action buttons in a row:
  - "Generate XAI" — `secondary` button (white bg, `brand-700` text, border)
  - "Download Clinical Report" — `primary` button (`brand-600` bg, white text)
- Buttons have subtle scale-up hover effect (`hover:scale-[1.02]` with transition)

### Footer
- 80px tall, `text-text-tertiary`, single line if possible
- Centered, all caps small text would be elegant: `text-xs uppercase tracking-wider`

## Animation Guidelines

- **Page load**: Fade in cards with stagger (50ms delay between cards). Use
  `framer-motion` if available, else CSS transitions on `opacity` + `translate-y`.
- **Hover**: 200ms ease for all colors and shadow changes
- **Card click → expand**: 300ms ease-out with subtle scale + opacity
- **Mode switch (segmentation ↔ XAI)**: Crossfade plotly canvas content over
  500ms. Sliders fade in/out as appropriate.
- **Don't overdo it.** Animations should feel snappy, not theatrical. If in
  doubt, less is more.

## Mobile Responsiveness

- Stack columns at < 768px breakpoint
- Reduce padding on small screens (24px → 16px)
- Make tabs scroll horizontally if they don't fit
- The 3D viewer should still be usable on tablet sizes (>= 768px). On phone
  sizes, show the data panels and a "View 3D" expand button instead of the
  full viewer.

## How To Approach The Merge

1. **Compare side by side.** For each component file in
   `frontend/components/`, find the corresponding file in `frontend-v0/components/`
   and identify what styling can be ported.
2. **Don't just copy v0's code.** v0 used shadcn/ui (Tailwind v4 + Radix
   primitives). Your stack is Tailwind v3 + plain HTML + minimal Radix. So
   port the *visual ideas* (colors, layout, structure) NOT the implementation.
3. **Use the `cn()` utility** from `clsx` + `tailwind-merge` for clean
   conditional classes — install if not present:
   ```bash
   npm install clsx tailwind-merge
   ```
4. **Test after each component.** Don't change all 9 at once. Pick one,
   refine it, save, hot reload, see how it looks, then proceed.

## Final Cleanup

After all components are upgraded:

1. Remove `frontend/V0_PROMPT.md` (no longer needed — design lives in code now)
2. Delete `frontend-v0/` folder entirely (no longer a reference — design is merged)
3. Update `frontend/README.md` with a "Design system" section listing the
   color tokens and typography scale
4. Verify the test flow:
   - Drag a BraTS folder → progress works → results load → 3D viewer renders →
     model switching works → XAI generates → PDF downloads

## Constraints Reminder

- **Don't break working logic.** Working things include: SSE+polling fallback,
  drag-and-drop, plotly mesh viewer, two-mode viewer architecture, all backend
  API integration.
- **Don't add new dependencies** unless absolutely needed. Inter font and
  framer-motion are acceptable. Anything else, justify first.
- **Stay on the current stack** (Next 14, React 18, Tailwind 3). Do not
  upgrade major versions.
- **Test on the actual case** (`BraTS-GLI-02143-102`) after every visual change
  to make sure nothing functional broke.

## Order Of Operations

```
1. git checkout -b feat/design-upgrade
2. Update tailwind.config.js with new color tokens
3. Update layout.tsx with Inter font
4. Refine globals.css (CSS variables, reset)
5. Component-by-component:
   a. UploadCard
   b. ProgressCard
   c. ModelPicker
   d. ViewerPanel
   e. MetricsPanel
   f. AnatomyCard
   g. EloquentCard
   h. AgreementBanner
   i. Header
6. Test full flow
7. Delete frontend-v0/ and V0_PROMPT.md
8. Update README
9. git commit -m "design: production-grade visual polish across dashboard"
10. git merge to dev
```

## Final Notes

The user has been working on this for months. They have strong taste even if
they don't always articulate it precisely. When they say "make it nicer," they
mean "make it feel like a real product, not a prototype." That's what this
brief targets.

If you're unsure about any single styling decision, choose the more restrained
option. Medical software shouldn't shout. It should communicate clearly,
quickly, and without distracting the user from the actual data.
