# Dashboard Empty State Redesign

**Date:** 2026-04-22
**Status:** Approved for implementation

---

## Problem

The current dashboard empty state misleads new users:
1. Primary CTA is "New Training" — step 4 of 4 in the actual flow, useless without connections first
2. "Dashboard" title appears twice (topbar + page H1)
3. The 4 preview cards are symmetric 2×2 which looks unnatural and doesn't reflect visual hierarchy
4. No guidance on what to do first
5. Sidebar items (Datasets, Model Studio, Training) are accessible even when prerequisites are missing

---

## Correct User Flow

```
Connections → Datasets → Model Studio → Training → Monitoring
```

- **Connections**: add data node endpoints; required before anything else
- **Datasets**: browse datasets retrieved from connected nodes
- **Model Studio**: pick an existing model or design one (Model Designer / ML Model Designer)
- **Training**: configure and launch federated learning
- **Monitoring**: the real-time dashboard, active job view, client status

Primary audience: researchers and medical scientists (not ops engineers). Guidance must be explicit and low-friction.

---

## Design Decisions

### 1. Header — Option B (no page header)

Remove the page-level header zone entirely. No H1 "Dashboard", no "Welcome back" as a separate block. The topbar already shows "Dashboard". The welcome message integrates into the setup banner.

**Before:**
```
[Topbar: Dashboard]
[Page H1: Dashboard]     [New Training] [Create Model]
[Welcome back, admin]
```

**After:**
```
[Topbar: Dashboard]
[Banner: Welcome back, Admin — start by adding a connection]  [Add Connection →]
```

Action buttons ("New Training", "Create Model") move to a ghost/disabled state below the banner while setup is incomplete.

### 2. Setup Banner — Style B (dots progress)

A compact dark-gradient banner that replaces the page header and communicates the current setup step.

**Structure:**
```
┌─────────────────────────────────────────────────────────┐
│  Welcome back, {user} — {step-specific message}   [CTA] │
│  ● ○ ○ ○   Step {n} of 4 — {step name}                 │
└─────────────────────────────────────────────────────────┘
```

- Background: `linear-gradient(100deg, #0F172A, #1E293B)`
- Active dot: `#0891B2`, width `20px`, height `4px`, border-radius `3px`
- Inactive dots: `#334155`, width `8px`, height `4px`
- Completed dots: `#059669`, width `8px`
- CTA button: `background: #0891B2`

**Step states:**

| Step | Condition | Banner message | CTA |
|------|-----------|----------------|-----|
| 1 | No connections | "start by adding your first data connection" | Add Connection → |
| 2 | Has connections, no dataset selected | "choose a dataset from your connected nodes" | Go to Datasets → |
| 3 | Has dataset, no model | "design or select a model in Model Studio" | Open Model Studio → |
| 4 | Has model, no training | "you're ready — launch your first training" | Start Training → |
| — | Has active/completed training | Banner hidden entirely | — |

**Detection logic (template context):**
- Step 1: `connections_count == 0`
- Step 2: `connections_count > 0 AND datasets_count == 0`
- Step 3: `datasets_count > 0 AND models_count == 0`
- Step 4: `models_count > 0 AND active_jobs_count == 0`
- Hidden: `active_jobs_count > 0`

### 3. Stat Cards

Four pills remain: Active Jobs, Models, Connections, Success Rate.

**Connections card special treatment when = 0:**
- Left border accent: `#D97706` (amber warning)
- Sub-label: "Add a connection to get started"
- Clickable — links to Connections page

All other zero-value cards: normal style, no special treatment.

### 4. Main Content — Layout Y

**Left (featured, ~60% width):** Active Trainings
- Dashed border (`#CBD5E1`), white background
- Header: small icon + "Active Trainings" label
- Description: "Real-time accuracy & loss curves from your federated training jobs will appear here"
- Ghost chart: muted bar chart silhouette (bars in `#E2E8F0`, opacity 0.5) with legend labels ("Accuracy", "Loss") in same muted color
- When jobs exist: replace entirely with real Chart.js training chart

**Right column (~40% width), top to bottom:**
1. **Recent Activity** — "Events from jobs, clients and models will appear as a live log" + `badge: No activity yet`
2. **Results & Models** — "Completed models ready to download and deploy" + `badge: No models yet`
3. **Client Status** — "Connected nodes and their health indicators" + `badge: No connections`

All right-column cards: white bg, `1px dashed #CBD5E1` border, 8px border-radius.

### 5. Sidebar — Disabled items when setup incomplete

When `connections_count == 0`, the following sidebar items render as visually disabled:
- Datasets
- Model Studio
- Training

**Disabled style:**
- Opacity: `0.4`
- Pointer events: `none` (non-clickable)
- Add a small lock icon (`data-lucide="lock"`, `width:10px`) after the label

Dashboard and Notifications sidebar items remain always active.

When `connections_count > 0` (step 2+), all items unlock normally.

### 6. Action buttons (post-setup context)

The "New Training" and "Create Model" buttons from the old header move to a secondary row below the banner, styled as ghost/disabled until prerequisites are met:
- Ghost style: `border: 1px dashed #E2E8F0`, `color: #94A3B8`, `cursor: not-allowed`
- Lock icon prefix when disabled
- Once all 4 steps complete (training exists), these become active with normal button styles

---

## What Does NOT Change

- The real-time monitoring view (shown when a job is active) — this is a separate template/state, not affected
- Chart.js integration, API polling, client cards — unchanged
- Design system tokens, sidebar structure, topbar — unchanged
- The "Pro Tip" card at the bottom — keep as-is, rewording is optional

---

## Template & CSS Scope

**Files to modify:**
- `MediNetHub/templates/webapp/dashboard_home.html` (or `dashboard.html` empty state branch) — banner, header removal, layout Y, stat card special treatment
- `MediNetHub/templates/base.html` — sidebar disabled state logic
- `MediNetHub/static/css/pages/dashboard.css` — ghost chart styles, layout Y, banner, disabled sidebar styles

**Context variables needed from view:**
- `connections_count` — integer
- `datasets_count` — integer  
- `models_count` — integer
- `active_jobs_count` — integer

If these aren't already passed, the view (`base_views.user_dashboard`) needs to query and pass them.
