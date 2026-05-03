# v0.dev Prompt for CranioVision Frontend

Paste this entire prompt into v0.dev to generate the visual design.
After v0 produces components, we adapt and wire them up to our backend.

---

# PROMPT TO PASTE INTO v0.dev

Build a clinical dashboard UI called "CranioVision" — a brain tumor MRI
analysis tool. Single page, dashboard layout, light theme. Audience is
radiologists and neurosurgeons. The aesthetic should feel like serious
medical software, not a generic SaaS product.

## Layout

Single page, dashboard style with a 12-column grid. Sections in this order:

### Header (full width)
- Logo placeholder + "CranioVision" wordmark in a serif/clinical font
- Tagline: "AI-assisted brain tumor segmentation and clinical analysis"
- Right side: small links for "Documentation" and "GitHub" (just placeholders)

### Top row (two cards, side by side, 50/50 split)
**Card 1 — Upload zone (left, 6 cols):**
- Title: "Upload MRI Case"
- Drag-and-drop area (dashed border, large, takes most of the card)
- Text: "Drop a BraTS case folder here, or click to browse"
- Below the drop zone: a dropdown labeled "Or select a demo case" with
  4 options: BraTS-GLI-02143-102, BraTS-GLI-02196-105, BraTS-GLI-02105-105,
  BraTS-GLI-02137-104

**Card 2 — Job status (right, 6 cols):**
- Title: "Analysis Progress"
- Large progress bar (gradient blue), percentage shown right-aligned
- Below the bar: current stage name + elapsed time
- Below that: a small list of stages with checkmarks (preprocessing,
  inference, ensemble, atlas registration, anatomy, done)
- When idle (no job running): show "Awaiting upload" centered placeholder

### Second row — model picker (full width, prominent)
- Title: "Choose Prediction"
- Tab bar with 4 buttons: "Attention U-Net" / "SwinUNETR" / "nnU-Net" /
  "Ensemble" (default selected = Ensemble, highlighted in blue)
- Below each tab: small badge showing mean Dice and total volume

### Main row — 3D viewer + metrics (8/4 split)
**3D Viewer (left, 8 cols, ~600px tall):**
- Title: "3D Anatomical Viewer"
- Large dark grey area placeholder where the 3D viewer will render
  (this is where Niivue will mount)
- Top toolbar of toggles:
  - "Segmentation" (toggle, on by default)
  - "Grad-CAM heatmap" (toggle, off by default, with "Generate" button
    next to it that says "Generate" until the user clicks)
- Right-side panel (within the viewer card) for opacity sliders:
  - "Segmentation opacity"
  - "Heatmap opacity"
- Bottom: small caption "Patient T1c MRI with model predictions overlaid.
  Use mouse to rotate, scroll to zoom."

**Metrics panel (right, 4 cols):**
- Title: "Volume Breakdown"
- Four large numbers stacked:
  - Total tumor: XXX cm³
  - Edema: XXX cm³
  - Enhancing tumor: XXX cm³
  - Necrotic core: XXX cm³
- Below: separator
- "Performance vs ground truth" subsection:
  - Mean Dice: 0.XXX
  - Whole tumor Dice: 0.XXX
  - Tumor core Dice: 0.XXX
  - Enhancing tumor Dice: 0.XXX

### Third row — anatomy + eloquent (6/6 split)

**Anatomical Context (left, 6 cols):**
- Title: "Anatomical Context"
- Subtitle: "Tumor location based on Harvard-Oxford atlas"
- Top half: a pie chart showing lobe distribution (use placeholder data:
  Frontal 50%, Subcortical 37%, Cingulate 6%, Other 7%)
- Bottom half: a small table titled "Top regions involved":
  - Region | % of tumor
  - 5 rows of placeholder data

**Eloquent Cortex Risk (right, 6 cols):**
- Title: "Eloquent Cortex Risk"
- Subtitle: "Surgical risk based on proximity to functional regions"
- 6 rows, one per region. Each row has:
  - Region name (left)
  - Distance in mm (middle)
  - Risk badge (right): pill-shaped, color-coded
    (red HIGH, orange MODERATE, yellow LOW, green MINIMAL)
- Regions to show:
  - Primary Motor Cortex — INVOLVED — HIGH (red)
  - Supplementary Motor Area — 1.4 mm — HIGH (red)
  - Primary Somatosensory Cortex — INVOLVED — HIGH (red)
  - Broca's Area — 12.7 mm — LOW (yellow)
  - Wernicke's Area — INVOLVED — HIGH (red)
  - Primary Visual Cortex — INVOLVED — HIGH (red)
- Below the table: a small alert box with a stethoscope icon:
  "Clinical recommendation: Pre-operative functional MRI and awake
  craniotomy planning recommended for HIGH-risk regions."

### Bottom row — agreement + actions (full width)
- Big agreement card with two columns:
  - Left: large stat "98.4%" + label "Multi-model unanimous"
  - Right: text "All 3 architectures agreed on 98.4% of voxels.
    Disagreement concentrated at tumor boundaries —
    radiologist review recommended at edge regions."
- Below: two action buttons in a row:
  - "Download Clinical Report (PDF)" — primary, blue, with download icon
  - "Generate XAI Explanation" — secondary, with brain icon

### Footer (full width)
- Small text:
  "CranioVision · Research use only — not for clinical diagnosis ·
   University of Moratuwa · 2026"
- Right side: "v1.0.0"

## Visual style requirements

- **Light theme.** White background (#FFFFFF), soft grey for panels (#F8F9FB),
  subtle borders (#E5E7EB).
- **Primary color:** clinical navy blue (#003366), accent blue (#0066CC).
- **Typography:** clean sans-serif (Inter or similar). Headings semi-bold.
  Numbers (volumes, Dice scores) in tabular monospace.
- **Spacing:** generous padding inside cards (24px). Card borders at 1px,
  subtle shadow on hover.
- **Risk badges:** red (#D73027), orange (#FC8D59), yellow (#FEE08B),
  green (#1A9850). Pill-shaped with white text, small padding.
- **Charts:** use Chart.js or Recharts, minimal styling, white background.
- **Icons:** lucide-react for any icons (download, upload, brain,
  stethoscope, info).
- **No animations** other than subtle hover states. No gradient backgrounds
  on cards. No glassmorphism. Conservative.

## Component breakdown

Generate as separate components:
1. `<UploadCard />` — drop zone + demo dropdown
2. `<ProgressCard />` — progress bar + stage list
3. `<ModelPickerTabs />` — 4 model tabs
4. `<ViewerPanel />` — 3D viewer placeholder + toolbars (mount point for Niivue)
5. `<MetricsPanel />` — volumes + Dice scores
6. `<AnatomyCard />` — pie chart + top regions table
7. `<EloquentCard />` — 6 risk rows + alert box
8. `<AgreementBanner />` — big stat + actions

Use TypeScript. Use Tailwind for styling. Use shadcn/ui components where
applicable (Card, Button, Tabs, Progress, Badge, Alert).

Make all data placeholders so I can wire them to real props later.
Don't include any data fetching logic.