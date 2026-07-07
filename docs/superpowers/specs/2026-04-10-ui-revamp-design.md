# MediNetHub UI Revamp - Design Specification

## Version
- **Date**: 2026-04-10
- **Status**: Approved
- **Branch**: v0.2
- **Approach**: Foundation First (design system → page-by-page migration)

---

## 1. Context & Problem Statement

MediNetHub is a Django-based federated learning platform for medical sciences. The current UI suffers from:

### Critical Issues
1. **No coherent design system** — 4 different primary color definitions across templates (`#2C3E50` in style.css, `#1976d2` in dashboard_home, `#0d6efd` in training, `#007bff` in home). Each template redefines its own `:root` CSS variables.
2. **Mixed languages** — Catalan ("Notificacions", "Veure totes"), Spanish ("Nombre de usuario"), and English ("Dashboard", "Login") mixed without pattern.
3. **~3500 lines of inline CSS** in individual templates vs 230 lines in the single global stylesheet. Causes duplication, inconsistencies, unmaintainable styles.

### High Severity
4. **Generic system typography** — `'Segoe UI', Tahoma, Geneva, Verdana, sans-serif` with no font pairing or type scale.
5. **Identical cards everywhere** — Same shadow, same radius, no elevation variation. Every page feels the same.
6. **Default Bootstrap navbar** — `navbar-dark bg-primary` with no branding, no logo, no identity.
7. **Home page without impact** — Text-only hero, FA icons at `fa-6x`, no illustrations, no social proof.

### Medium Severity
8. No dark mode for a technical tool used for extended sessions.
9. No breadcrumbs or consistent workflow context across pages.
10. Footer with broken links (Documentation → #, API → #, Support → #).
11. Login/Register pages without personality or branding.
12. Model Designer visually disconnected from the rest of the app.

---

## 2. Design Decisions

### 2.1 Visual Style
- **Style**: Tech-Scientific Modern, light tone
- **Personality**: Gradients on accents, rounded corners (12-14px), subtle decorative elements on cards, translucent status badges, strong typography hierarchy
- **No purple** — explicitly excluded from palette
- **Reference platforms**: Weights & Biases, Neptune.ai, Grafana

### 2.2 Color Palette — Navy + Teal

Evolved from the existing `#2C3E50` / `#3498DB` base, modernized with more contrast.

| Token | Value | Usage |
|-------|-------|-------|
| `--color-primary` | `#0F172A` | Text headings, sidebar background, primary surfaces |
| `--color-secondary` | `#1E40AF` | Primary actions, links, active states |
| `--color-accent` | `#0891B2` | Accent elements, gradients endpoint, highlights |
| `--color-success` | `#059669` | Success states, running indicators, positive metrics |
| `--color-warning` | `#D97706` | Warning states, pending indicators |
| `--color-danger` | `#DC2626` | Error states, destructive actions |
| `--color-bg` | `#F8FAFC` | Page background |
| `--color-surface` | `#FFFFFF` | Cards, panels |
| `--color-border` | `#E2E8F0` | Card borders, dividers |
| `--color-text` | `#0F172A` | Primary text |
| `--color-text-secondary` | `#64748B` | Secondary text, labels |
| `--color-text-muted` | `#94A3B8` | Muted text, placeholders |

**Gradient tokens:**
| Token | Value | Usage |
|-------|-------|-------|
| `--gradient-primary` | `linear-gradient(135deg, #1E40AF, #0891B2)` | Logo, primary CTAs, nav active bg |
| `--gradient-success` | `linear-gradient(90deg, #059669, #34D399)` | Success progress bars |
| `--gradient-warning` | `linear-gradient(90deg, #D97706, #FBBF24)` | Warning progress bars |
| `--gradient-accent` | `linear-gradient(90deg, #0891B2, #22D3EE)` | Accent progress bars |

**Status badge pattern:** Translucent background of semantic color at 12% opacity + solid text.
```css
.status-running { background: rgba(5, 150, 105, 0.12); color: #059669; }
.status-pending { background: rgba(37, 99, 235, 0.12); color: #2563EB; }
.status-completed { background: rgba(100, 116, 139, 0.12); color: #64748B; }
.status-failed { background: rgba(220, 38, 38, 0.12); color: #DC2626; }
```

**Note:** This palette is the starting point. We will iterate during implementation.

### 2.3 Typography

| Role | Font | Weight | Size |
|------|------|--------|------|
| Headings | Inter | 700 | 24/20/18/16px |
| Body | Inter | 400 | 14px |
| Labels | Inter | 600 | 12px (uppercase, letter-spacing: 0.5px) |
| Metric values | Inter | 700 | 28px (stat cards), 22px (inline) |
| Code/Logs | JetBrains Mono | 400 | 13px |

**Type scale:** 12 / 13 / 14 / 16 / 18 / 20 / 24 / 28 / 32px

**Google Fonts import:**
```
Inter:wght@400;500;600;700
JetBrains Mono:wght@400;500
```

### 2.4 Language
- **English only** throughout the entire UI
- All existing Catalan and Spanish strings will be replaced
- Consistent terminology: "Training Job" (not "Entrenamiento"), "Datasets" (not "Conjuntos de datos"), etc.

### 2.5 Navigation — Collapsible Sidebar

**Structure:**
- **Collapsed**: 60px wide, icon-only
- **Expanded**: 220px wide, icons + labels
- **Toggle**: Click on expand/collapse arrow at bottom
- **Background**: `#0F172A` (primary dark)
- **Active item**: Teal pill background (`rgba(8, 145, 178, 0.15)`) + teal icon color
- **Inactive items**: `#64748B` icons

**Sidebar items (authenticated):**
1. Dashboard
2. Datasets
3. Model Studio
4. Training
5. ---separator---
6. Notifications (with badge)
7. ---spacer---
8. Profile (bottom, avatar)
9. Collapse toggle (bottom)

**Top bar (minimal):**
- Page title + breadcrumb on the left
- Notification bell + user avatar on the right
- Background: white, border-bottom subtle

**Pre-auth pages (Home, Login, Register):**
- No sidebar
- Independent minimal layout with centered branding

**Model Studio adjustment:**
- Current filter sidebar (col-md-3) moves to horizontal toolbar with dropdown filters
- Eliminates double-sidebar conflict

### 2.6 Platform
- **Desktop-only** — no mobile responsive effort
- Minimum viewport: 1280px
- Optimized for 1440px-1920px

### 2.7 Card System

Three elevation levels:
| Level | Usage | Shadow |
|-------|-------|--------|
| Flat | Table containers, panels | `border: 1px solid var(--color-border)` |
| Raised | Stat cards, content cards | `box-shadow: 0 2px 8px rgba(0,0,0,0.06)` |
| Elevated | Modals, dropdowns | `box-shadow: 0 8px 24px rgba(0,0,0,0.12)` |

- Border-radius: 12px for cards, 8px for inputs/badges, 6px for status pills
- Decorative gradient circles in top-right corner of stat cards (subtle, 10-15% opacity)

### 2.8 Icon System
- **Library**: Lucide Icons (SVG, consistent 2px stroke)
- **Replaces**: Font Awesome throughout
- No emojis as icons anywhere

### 2.9 Bootstrap Relationship
- **Keep Bootstrap 5** as the base framework (grid, utilities, JS components like modals/dropdowns/tooltips)
- **Override** Bootstrap's default colors, typography, and component styles via our design system tokens
- **Do not add** new Bootstrap dependencies; our custom CSS takes precedence
- **Gradually reduce** reliance on Bootstrap-specific classes where our components provide the same functionality

---

## 3. Scope

### In Scope — Phase 1: Visual Revamp
1. Global design system CSS (tokens, components, layout)
2. Sidebar + top bar base layout
3. Migrate all existing pages to new design system:
   - Home (public landing)
   - Login / Register
   - Dashboard Home
   - Datasets + Dataset Details
   - Model Studio
   - Current Model Designer (visual refresh, not rebuild)
   - Training + Training Dashboard
   - Job Detail / Experiment Detail
   - Notifications
   - Profile
4. Unify all language to English
5. Remove inline CSS from templates, consolidate into stylesheet modules
6. Enhanced Model Download UI (new page/components)

### In Scope — Phase 2: New Model Designer
- Will be specified separately after Phase 1 completes
- User will provide requirements

### Out of Scope
- Mobile responsive design
- Dark mode (future consideration, but tokens will be structured to support it)
- Flower framework migration
- Additional federated strategies UI (blocked by Flower migration)
- i18n system

---

## 4. Implementation Strategy — Foundation First

### Phase 1: Design System Foundation
- Create `static/css/design-system.css` with all tokens, reset, base typography
- Create `static/css/components.css` with reusable component classes
- Create `static/css/layout.css` with sidebar + content area layout
- Import Inter + JetBrains Mono fonts
- Integrate Lucide Icons (CDN or local)

### Phase 2: Base Layout
- New `base.html` with sidebar layout (authenticated)
- New `base_public.html` for pre-auth pages
- Minimal top bar component
- Toast/notification system migration

### Phase 3: Page Migration (one by one)
Order by visibility and complexity:
1. Dashboard Home — highest visibility, establishes patterns
2. Login / Register — simple, validates auth layout
3. Home (public landing) — validates public layout
4. Training list — table patterns
5. Training Dashboard — metric/chart patterns
6. Datasets — card list + stepper patterns
7. Model Studio — grid + filter patterns (toolbar migration)
8. Model Designer — visual refresh only (not rebuild)
9. Job Detail / Experiment Detail — detail page patterns
10. Dataset Details — tab/metric patterns
11. Notifications — simple list pattern
12. Profile — form/settings pattern

### Phase 4: New Features UI
- Enhanced Model Download system
- Any additional v0.2 UI work

### Phase 5: New Model Designer
- Separate spec after Phase 1-4 complete

---

## 5. File Structure (Target)

```
static/
  css/
    design-system.css    # Tokens, reset, typography, colors
    components.css       # Buttons, cards, badges, tables, forms
    layout.css           # Sidebar, top bar, content area, grid
    pages/
      dashboard.css      # Dashboard-specific styles (minimal)
      training.css       # Training-specific styles (minimal)
      model-designer.css # Model designer styles
      ...
  js/
    main.js              # Global JS (toasts, notifications)
    sidebar.js           # Sidebar collapse/expand logic
templates/
  base.html              # Authenticated layout (sidebar)
  base_public.html       # Public layout (no sidebar)
  webapp/
    ...                  # Individual page templates (minimal inline CSS)
```

---

## 6. Quality Criteria

- Zero inline `<style>` blocks in templates (all CSS in external files)
- Single `:root` definition in design-system.css (no per-template overrides)
- All text in English
- Consistent component usage across all pages
- WCAG AA contrast compliance (4.5:1 minimum for text)
- Lucide icons only (no Font Awesome, no emojis)
- All colors via CSS custom properties (no hardcoded hex in templates)
