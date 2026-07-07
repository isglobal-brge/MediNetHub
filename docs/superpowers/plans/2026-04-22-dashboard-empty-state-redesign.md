# Dashboard Empty State Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dashboard empty state with a contextual onboarding banner + asymmetric layout that guides users through the correct Connections → Datasets → Model → Training flow, removes the duplicate "Dashboard" title, and locks sidebar items when no connections exist.

**Architecture:** A new context processor injects `global_connections_count` into every template (one lightweight DB query per request) to power the sidebar lock in `base.html`. The `user_dashboard` view gets `datasets_count` added to its context. `dashboard_home.html` is restructured (not rewritten from scratch — targeted edits). CSS additions go into existing `dashboard.css` and `layout.css`.

**Tech Stack:** Django 5.2 templates, Bootstrap 5 grid (`col-lg-7` / `col-lg-5`), Lucide icons (already loaded), CSS custom properties (`--color-accent: #0891B2`, `--color-primary: #0F172A`), no new JavaScript dependencies.

---

## File Map

| Action | File | What changes |
|--------|------|--------------|
| Create | `MediNetHub/webapp/context_processors.py` | `connections_status()` processor |
| Modify | `MediNetHub/medinet/settings.py` | Register context processor |
| Modify | `MediNetHub/webapp/base_views.py` | Add `datasets_count` to context |
| Modify | `MediNetHub/static/css/layout.css` | `.sidebar-item--locked` styles |
| Modify | `MediNetHub/static/css/pages/dashboard.css` | Banner, layout-Y, ghost chart, disabled button styles |
| Modify | `MediNetHub/templates/base.html` | Sidebar locked state |
| Modify | `MediNetHub/templates/webapp/dashboard_home.html` | Full redesign |

---

## Task 1: Context Processor — global connections count

**Files:**
- Create: `MediNetHub/webapp/context_processors.py`
- Modify: `MediNetHub/medinet/settings.py` lines 104–109

- [ ] **Step 1.1: Create the context processor file**

```python
# MediNetHub/webapp/context_processors.py
from .models import Connection


def connections_status(request):
    if not request.user.is_authenticated:
        return {'global_connections_count': 0}
    count = Connection.objects.filter(user=request.user).count()
    return {'global_connections_count': count}
```

- [ ] **Step 1.2: Register it in settings**

In `MediNetHub/medinet/settings.py`, find the `context_processors` list inside `TEMPLATES` and add the new processor:

```python
'context_processors': [
    'django.template.context_processors.debug',
    'django.template.context_processors.request',
    'django.contrib.auth.context_processors.auth',
    'django.contrib.messages.context_processors.messages',
    'webapp.context_processors.connections_status',   # ← add this line
],
```

- [ ] **Step 1.3: Verify the processor loads without error**

Start the Django dev server and open any page. If it loads without a 500 error, the processor is registered correctly.

```bash
python manage.py runserver
```

Expected: server starts, no `ImportError` or `TemplateSyntaxError`.

- [ ] **Step 1.4: Commit**

```bash
git add MediNetHub/webapp/context_processors.py MediNetHub/medinet/settings.py
git commit -m "feat: add connections_status context processor for global sidebar lock"
```

---

## Task 2: Add datasets_count to dashboard view context

**Files:**
- Modify: `MediNetHub/webapp/base_views.py` lines 213–228

- [ ] **Step 2.1: Query datasets count and add to context**

In `base_views.py`, after the `connections_count` block (around line 182) and before the `stats` dict, add:

```python
# Get datasets count
datasets_count = Dataset.objects.filter(user=request.user).count()
```

Then add it to the `stats` dict (lines 214–219):

```python
stats = {
    'total_models': models_count,
    'total_jobs': total_jobs,
    'active_connections': connections_count,
    'success_rate': success_rate,
    'datasets_count': datasets_count,       # ← add this line
}
```

- [ ] **Step 2.2: Verify the view still renders**

Open `http://localhost:8000/panel/` — page should load without error.

- [ ] **Step 2.3: Commit**

```bash
git add MediNetHub/webapp/base_views.py
git commit -m "feat: add datasets_count to dashboard view context"
```

---

## Task 3: CSS — banner, layout-Y, ghost chart, disabled buttons, sidebar lock

**Files:**
- Modify: `MediNetHub/static/css/layout.css` (append)
- Modify: `MediNetHub/static/css/pages/dashboard.css` (append)

- [ ] **Step 3.1: Add sidebar locked state to layout.css**

Append to the end of `MediNetHub/static/css/layout.css`:

```css
/* ===== Sidebar locked item (no connections) ===== */
.sidebar-item--locked {
  opacity: 0.38;
  pointer-events: none;
  cursor: not-allowed;
}

.sidebar-item--locked .sidebar-lock-icon {
  width: 10px;
  height: 10px;
  margin-left: auto;
  flex-shrink: 0;
}
```

- [ ] **Step 3.2: Add dashboard-specific styles to dashboard.css**

Append to the end of `MediNetHub/static/css/pages/dashboard.css`:

```css
/* ===== Setup Banner (Banner B) ===== */
.setup-banner {
  background: linear-gradient(100deg, #0F172A 0%, #1E293B 100%);
  padding: 14px 24px;
  border-radius: 0;
  margin-bottom: 0;
}

.setup-banner__row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}

.setup-banner__title {
  font-size: var(--text-sm);
  font-weight: 600;
  color: #fff;
  margin: 0 0 2px;
}

.setup-banner__sub {
  font-size: var(--text-xs);
  color: #64748B;
  margin: 0;
}

.setup-banner__btn {
  background: var(--color-accent);
  color: #fff;
  border: none;
  border-radius: var(--radius-md);
  padding: 7px 16px;
  font-size: var(--text-xs);
  font-weight: 600;
  text-decoration: none;
  white-space: nowrap;
  transition: opacity 0.15s;
}

.setup-banner__btn:hover {
  opacity: 0.88;
  color: #fff;
}

.setup-banner__progress {
  display: flex;
  align-items: center;
  gap: 5px;
}

.setup-banner__dot {
  height: 4px;
  border-radius: 3px;
  background: #334155;
  width: 8px;
  transition: all 0.2s;
}

.setup-banner__dot--active {
  width: 20px;
  background: var(--color-accent);
}

.setup-banner__dot--done {
  background: var(--color-success);
}

.setup-banner__step-label {
  font-size: 11px;
  color: #475569;
  margin-left: 6px;
}

.setup-banner__step-label span {
  color: #7DD3FC;
  font-weight: 600;
}

/* ===== Layout Y — main content grid ===== */
.dashboard-layout-y {
  display: flex;
  gap: 1.5rem;
  align-items: flex-start;
}

.dashboard-layout-y__featured {
  flex: 1.4;
  min-width: 0;
}

.dashboard-layout-y__stack {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  min-width: 0;
}

@media (max-width: 991px) {
  .dashboard-layout-y {
    flex-direction: column;
  }
}

/* ===== Ghost chart (empty state Active Trainings) ===== */
.ghost-chart {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 16px;
}

.ghost-chart__bars {
  display: flex;
  align-items: flex-end;
  gap: 4px;
  height: 56px;
}

.ghost-chart__bar {
  flex: 1;
  border-radius: 3px 3px 0 0;
  background: #E2E8F0;
  opacity: 0.55;
}

.ghost-chart__legend {
  display: flex;
  gap: 14px;
}

.ghost-chart__legend-item {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: #CBD5E1;
}

.ghost-chart__legend-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #E2E8F0;
}

/* ===== Preview cards (right column empty state) ===== */
.preview-card {
  background: var(--surface);
  border: 1px dashed #CBD5E1;
  border-radius: var(--radius-lg);
  padding: 14px 16px;
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.preview-card__icon {
  width: 32px;
  height: 32px;
  border-radius: var(--radius-md);
  background: #F1F5F9;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  color: #94A3B8;
}

.preview-card__title {
  font-size: var(--text-sm);
  font-weight: 600;
  color: #64748B;
  margin: 0 0 3px;
}

.preview-card__desc {
  font-size: var(--text-xs);
  color: #CBD5E1;
  margin: 0 0 6px;
  line-height: 1.4;
}

.preview-card__badge {
  display: inline-block;
  font-size: 10px;
  padding: 2px 8px;
  background: #F1F5F9;
  color: #94A3B8;
  border-radius: 10px;
}

/* ===== Connections stat card — zero state ===== */
.stat-card--connections-empty {
  border-left: 3px solid var(--color-warning) !important;
}

.stat-card--connections-empty .stat-cta {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 10px;
  color: var(--color-accent);
  text-decoration: none;
  font-weight: 500;
  margin-top: 4px;
}

.stat-card--connections-empty .stat-cta:hover {
  text-decoration: underline;
}

/* ===== Ghost action buttons (header actions when locked) ===== */
.btn-ghost-locked {
  border: 1px dashed #E2E8F0;
  color: #94A3B8;
  background: transparent;
  border-radius: var(--radius-md);
  padding: 7px 16px;
  font-size: var(--text-sm);
  cursor: not-allowed;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  pointer-events: none;
}
```

- [ ] **Step 3.3: Verify CSS loads without errors**

Open browser DevTools → Console. No CSS parse errors should appear.

- [ ] **Step 3.4: Commit**

```bash
git add MediNetHub/static/css/layout.css MediNetHub/static/css/pages/dashboard.css
git commit -m "feat: add banner, layout-Y, ghost chart and locked sidebar CSS"
```

---

## Task 4: base.html — sidebar locked state

**Files:**
- Modify: `MediNetHub/templates/base.html` (sidebar nav section, ~lines 40–95)

- [ ] **Step 4.1: Read the current sidebar nav items**

Open `base.html` and find the three sidebar items for Datasets, Model Studio, and Training. They look like:

```html
<a href="{% url 'datasets' %}"
   class="sidebar-item {% if request.resolver_match.url_name == 'datasets' ... %}active{% endif %}">
  <i data-lucide="database"></i>
  <span class="sidebar-item-label">Datasets</span>
</a>
```

- [ ] **Step 4.2: Add locked class and lock icon to Datasets, Model Studio, Training**

For each of the three items (`datasets`, `model_studio`/`model_designer`, `training`), add the locked class and lock icon when `global_connections_count == 0`:

```html
<!-- Datasets -->
<a href="{% url 'datasets' %}"
   class="sidebar-item {% if request.resolver_match.url_name == 'datasets' or request.resolver_match.url_name == 'dataset_details' %}active{% endif %} {% if global_connections_count == 0 %}sidebar-item--locked{% endif %}">
  <i data-lucide="database"></i>
  <span class="sidebar-item-label">Datasets</span>
  {% if global_connections_count == 0 %}
    <i data-lucide="lock" class="sidebar-lock-icon"></i>
  {% endif %}
</a>

<!-- Model Studio — apply same pattern -->
<a href="{% url 'model_studio' %}"
   class="sidebar-item {% if request.resolver_match.url_name in 'model_studio,model_designer,...' %}active{% endif %} {% if global_connections_count == 0 %}sidebar-item--locked{% endif %}">
  <i data-lucide="boxes"></i>
  <span class="sidebar-item-label">Model Studio</span>
  {% if global_connections_count == 0 %}
    <i data-lucide="lock" class="sidebar-lock-icon"></i>
  {% endif %}
</a>

<!-- Training — apply same pattern -->
<a href="{% url 'training' %}"
   class="sidebar-item {% if request.resolver_match.url_name in 'training,...' %}active{% endif %} {% if global_connections_count == 0 %}sidebar-item--locked{% endif %}">
  <i data-lucide="play-circle"></i>
  <span class="sidebar-item-label">Training</span>
  {% if global_connections_count == 0 %}
    <i data-lucide="lock" class="sidebar-lock-icon"></i>
  {% endif %}
</a>
```

> **Note:** Copy the exact `active` class condition from the current `base.html` for each item — do not guess the URL names. Only add the `sidebar-item--locked` class and lock icon.

- [ ] **Step 4.3: Call `lucide.createIcons()` covers the new lock icons**

The lock icons use `data-lucide="lock"`. Lucide is initialized in `base.html` at the bottom via `lucide.createIcons()`. Verify that call exists — no action needed if it does.

- [ ] **Step 4.4: Visual check — no connections**

With no connections in the DB, open any page. The Datasets, Model Studio, and Training sidebar items should appear faded (opacity ~0.38) with a small lock icon, and clicking them should do nothing.

- [ ] **Step 4.5: Commit**

```bash
git add MediNetHub/templates/base.html
git commit -m "feat: lock sidebar items when no connections exist"
```

---

## Task 5: dashboard_home.html — full redesign

**Files:**
- Modify: `MediNetHub/templates/webapp/dashboard_home.html`

This is the main task. Make the following targeted edits to the existing template.

### 5a: Remove page header, add setup banner

- [ ] **Step 5a.1: Replace the page header block**

Find and replace the entire `<!-- ===== Page Header ===== -->` section (lines 14–30):

```html
<!-- REMOVE THIS ENTIRE BLOCK:
<div class="dashboard-page-header">
  <div>
    <h2 class="page-title">Dashboard</h2>
    <p class="page-subtitle">Welcome back, {{ user.first_name|default:user.username }}</p>
  </div>
  <div class="header-actions">
    <a href="{% url 'training' %}" class="btn btn-gradient px-4">...</a>
    <a href="{% url 'model_studio' %}" class="btn btn-outline-secondary-custom px-4">...</a>
  </div>
</div>
-->
```

Replace with the setup banner. Determine the current step using the context variables, then render the correct banner message and dot states:

```html
<!-- ===== Setup Banner ===== -->
{% if stats.active_connections == 0 %}
  {% with step=1 step_name="Connections" banner_msg="start by adding your first data connection" banner_url=datasets_url banner_label="Add Connection" %}
  {% include "webapp/partials/setup_banner.html" %}
  {% endwith %}
{% elif stats.datasets_count == 0 %}
  {% with step=2 step_name="Datasets" banner_msg="choose a dataset from your connected nodes" banner_url=datasets_url banner_label="Go to Datasets" %}
  {% include "webapp/partials/setup_banner.html" %}
  {% endwith %}
{% elif stats.total_models == 0 %}
  {% with step=3 step_name="Model" banner_msg="design or select a model in Model Studio" banner_url=model_studio_url banner_label="Open Model Studio" %}
  {% include "webapp/partials/setup_banner.html" %}
  {% endwith %}
{% elif active_jobs|length == 0 %}
  {% with step=4 step_name="Training" banner_msg="you're ready — launch your first training" banner_url=training_url banner_label="Start Training" %}
  {% include "webapp/partials/setup_banner.html" %}
  {% endwith %}
{% endif %}
<!-- Banner hidden when active_jobs|length > 0 — user is past onboarding -->
```

> **Simpler alternative (avoid partial):** Inline the banner directly with `{% if/elif %}` blocks for each step, duplicating the HTML structure. This avoids creating a partial file and is clearer for a single-use component. Use this approach if the partial adds complexity.

**Recommended simpler approach — inline the banner:**

```html
<!-- ===== Setup Banner (hidden once user has active jobs) ===== -->
{% if active_jobs|length == 0 %}
<div class="setup-banner">
  <div class="setup-banner__row">
    <div>
      {% if stats.active_connections == 0 %}
        <p class="setup-banner__title">Welcome back, {{ user.first_name|default:user.username }} — start by adding your first data connection</p>
        <p class="setup-banner__sub">Step 1 of 4 to launch your first federated training</p>
      {% elif stats.datasets_count == 0 %}
        <p class="setup-banner__title">Connection added — now choose a dataset from your nodes</p>
        <p class="setup-banner__sub">Step 2 of 4 — browse available datasets</p>
      {% elif stats.total_models == 0 %}
        <p class="setup-banner__title">Dataset ready — design or pick a model in Model Studio</p>
        <p class="setup-banner__sub">Step 3 of 4 — build your federated model architecture</p>
      {% else %}
        <p class="setup-banner__title">All set — launch your first federated training</p>
        <p class="setup-banner__sub">Step 4 of 4 — configure and start training</p>
      {% endif %}
    </div>
    {% if stats.active_connections == 0 %}
      <a href="{% url 'datasets' %}" class="setup-banner__btn">Add Connection →</a>
    {% elif stats.datasets_count == 0 %}
      <a href="{% url 'datasets' %}" class="setup-banner__btn">Go to Datasets →</a>
    {% elif stats.total_models == 0 %}
      <a href="{% url 'model_studio' %}" class="setup-banner__btn">Open Model Studio →</a>
    {% else %}
      <a href="{% url 'training' %}" class="setup-banner__btn">Start Training →</a>
    {% endif %}
  </div>

  <!-- Progress dots -->
  <div class="setup-banner__progress">
    <div class="setup-banner__dot {% if stats.active_connections > 0 %}setup-banner__dot--done{% else %}setup-banner__dot--active{% endif %}"></div>
    <div class="setup-banner__dot {% if stats.datasets_count > 0 and stats.active_connections > 0 %}setup-banner__dot--done{% elif stats.active_connections > 0 and stats.datasets_count == 0 %}setup-banner__dot--active{% endif %}"></div>
    <div class="setup-banner__dot {% if stats.total_models > 0 %}setup-banner__dot--done{% elif stats.active_connections > 0 and stats.datasets_count > 0 and stats.total_models == 0 %}setup-banner__dot--active{% endif %}"></div>
    <div class="setup-banner__dot {% if active_jobs|length > 0 %}setup-banner__dot--done{% elif stats.total_models > 0 and active_jobs|length == 0 %}setup-banner__dot--active{% endif %}"></div>
    <span class="setup-banner__step-label">
      Step
      {% if stats.active_connections == 0 %}<span>1</span>
      {% elif stats.datasets_count == 0 %}<span>2</span>
      {% elif stats.total_models == 0 %}<span>3</span>
      {% else %}<span>4</span>{% endif %}
      of 4 —
      {% if stats.active_connections == 0 %}Connections
      {% elif stats.datasets_count == 0 %}Datasets
      {% elif stats.total_models == 0 %}Model Studio
      {% else %}Training{% endif %}
    </span>
  </div>
</div>
{% endif %}
```

### 5b: Connections stat card — zero state treatment

- [ ] **Step 5b.1: Update the Connections stat card**

Find the Connections stat card block (lines 64–75) and replace with:

```html
<!-- Connections -->
<div class="col-6 col-lg-3">
  <div class="stat-card stat-card--warning animate-in {% if stats.active_connections == 0 %}stat-card--connections-empty{% endif %}">
    <p class="stat-label">Connections</p>
    <a href="{% url 'datasets' %}" class="stat-value text-decoration-none">{{ stats.active_connections }}</a>
    {% if stats.active_connections == 0 %}
      <a href="{% url 'datasets' %}" class="stat-cta">
        <i data-lucide="plus-circle" style="width:11px;height:11px;"></i>
        Add a connection
      </a>
    {% endif %}
    <div class="progress-custom stat-bar">
      <div class="progress-bar progress-bar--warning"
           role="progressbar"
           style="width: 45%; min-width: 8px;"></div>
    </div>
  </div>
</div>
```

### 5c: Rework main content to Layout Y

- [ ] **Step 5c.1: Replace the entire `<!-- ===== Main Content Row ===== -->` section**

Remove the current two-column Bootstrap row (lines 95–373) and replace with:

```html
<!-- ===== Main Content — Layout Y ===== -->
<div class="dashboard-layout-y">

  <!-- Featured left: Active Trainings -->
  <div class="dashboard-layout-y__featured">
    <div class="card-raised mb-4">
      <div class="card-header-custom d-flex align-items-center justify-content-between">
        <div class="d-flex align-items-center gap-2">
          <i data-lucide="play-circle" style="width:16px;height:16px;color:var(--color-accent);"></i>
          <span>Active Trainings</span>
        </div>
        <a href="{% url 'training' %}" class="btn btn-sm btn-outline-secondary-custom">View All</a>
      </div>

      <div class="p-4">
        {% if active_jobs %}
          {% for job in active_jobs %}
          <div class="activity-item animate-in">
            <div class="d-flex align-items-start justify-content-between">
              <div class="flex-grow-1">
                <div class="mb-2">
                  {% if job.status == 'running' %}
                    <span class="status-badge status-running">Running</span>
                  {% elif job.status == 'pending' %}
                    <span class="status-badge status-pending">Pending</span>
                  {% else %}
                    <span class="status-badge status-server-ready">Ready</span>
                  {% endif %}
                </div>
                <h6 class="job-title">{{ job.name }}</h6>
                <div class="progress-custom mb-2">
                  <div class="progress-bar progress-bar--success"
                       role="progressbar"
                       style="width: {{ job.progress }}%; min-width: 2px;"
                       aria-valuenow="{{ job.progress }}"
                       aria-valuemin="0" aria-valuemax="100"></div>
                </div>
                <div class="job-meta">
                  <i data-lucide="bar-chart-3"></i>
                  <span>Round {{ job.current_round }}/{{ job.total_rounds }}</span>
                  {% if job.progress > 0 %}
                    <span>&bull;</span>
                    <span>{{ job.progress }}% complete</span>
                  {% endif %}
                </div>
              </div>
              <div class="ms-3 flex-shrink-0">
                <a href="{% url 'dashboard' job.id %}"
                   class="btn btn-sm btn-outline-secondary-custom"
                   title="View training">
                  <i data-lucide="eye" style="width:14px;height:14px;"></i>
                </a>
              </div>
            </div>
          </div>
          {% endfor %}

        {% else %}
          <!-- Empty state with ghost chart -->
          <div style="padding: 8px 0 4px;">
            <p style="font-size:var(--text-sm);color:#94A3B8;margin:0 0 4px;">
              Real-time accuracy &amp; loss curves from your federated training jobs will appear here
            </p>
            <!-- Ghost chart -->
            <div class="ghost-chart">
              <div class="ghost-chart__bars">
                <div class="ghost-chart__bar" style="height:28%"></div>
                <div class="ghost-chart__bar" style="height:42%"></div>
                <div class="ghost-chart__bar" style="height:38%"></div>
                <div class="ghost-chart__bar" style="height:58%"></div>
                <div class="ghost-chart__bar" style="height:52%"></div>
                <div class="ghost-chart__bar" style="height:68%"></div>
                <div class="ghost-chart__bar" style="height:63%"></div>
                <div class="ghost-chart__bar" style="height:78%"></div>
                <div class="ghost-chart__bar" style="height:73%"></div>
                <div class="ghost-chart__bar" style="height:84%"></div>
                <div class="ghost-chart__bar" style="height:88%"></div>
                <div class="ghost-chart__bar" style="height:85%"></div>
              </div>
              <div class="ghost-chart__legend">
                <div class="ghost-chart__legend-item">
                  <div class="ghost-chart__legend-dot"></div>
                  Accuracy
                </div>
                <div class="ghost-chart__legend-item">
                  <div class="ghost-chart__legend-dot"></div>
                  Loss
                </div>
              </div>
            </div>
          </div>
        {% endif %}
      </div>
    </div><!-- /Active Trainings -->

    <!-- Pro Tip -->
    {% if active_jobs|length < 3 %}
    <div class="pro-tip-card animate-in">
      <div class="tip-header">
        <i data-lucide="lightbulb"></i>
        Pro Tip
      </div>
      <blockquote>
        <p id="tip-quote" class="tip-quote"></p>
        <footer id="tip-caption" class="tip-caption"></footer>
      </blockquote>
    </div>
    {% endif %}
  </div><!-- /Featured left -->

  <!-- Right column stack: Recent Activity → Results → Client Status -->
  <div class="dashboard-layout-y__stack">

    <!-- 1. Recent Activity -->
    <div class="preview-card">
      <div class="preview-card__icon">
        <i data-lucide="clock" style="width:16px;height:16px;"></i>
      </div>
      <div class="flex-grow-1 min-w-0">
        <p class="preview-card__title">Recent Activity</p>
        {% if recent_jobs %}
          {% for job in recent_jobs|slice:":3" %}
          <div class="activity-item animate-in" style="border-bottom: 1px solid var(--border); padding: 6px 0; margin-bottom:0;">
            <div class="d-flex align-items-start gap-2">
              <div class="flex-grow-1">
                <div class="mb-1">
                  {% if job.status == 'completed' %}
                    <span class="status-badge status-completed">Completed</span>
                  {% elif job.status == 'failed' %}
                    <span class="status-badge status-failed">Failed</span>
                  {% elif job.status == 'running' %}
                    <span class="status-badge status-running">Running</span>
                  {% else %}
                    <span class="status-badge status-pending">Pending</span>
                  {% endif %}
                </div>
                <p class="job-title mb-0" style="font-size:var(--text-xs);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{{ job.name }}</p>
                <div class="job-meta" style="font-size:10px;">
                  <i data-lucide="clock" style="width:10px;height:10px;"></i>
                  <span>{{ job.created_at|timesince }} ago</span>
                </div>
              </div>
            </div>
          </div>
          {% endfor %}
        {% else %}
          <p class="preview-card__desc">Events from jobs, clients and models will appear here as a live log</p>
          <span class="preview-card__badge">No activity yet</span>
        {% endif %}
      </div>
    </div>

    <!-- 2. Results & Models -->
    <div class="preview-card">
      <div class="preview-card__icon">
        <i data-lucide="check-circle" style="width:16px;height:16px;"></i>
      </div>
      <div>
        <p class="preview-card__title">Results &amp; Models</p>
        {% if stats.total_models > 0 %}
          <p class="preview-card__desc">
            <a href="{% url 'models_list' %}" style="color:var(--color-accent);font-weight:600;">
              {{ stats.total_models }} model{{ stats.total_models|pluralize }} ready
            </a>
          </p>
        {% else %}
          <p class="preview-card__desc">Completed models ready to download and deploy</p>
          <span class="preview-card__badge">No models yet</span>
        {% endif %}
      </div>
    </div>

    <!-- 3. Client Status -->
    <div class="preview-card">
      <div class="preview-card__icon">
        <i data-lucide="users" style="width:16px;height:16px;"></i>
      </div>
      <div>
        <p class="preview-card__title">Client Status</p>
        {% if stats.active_connections > 0 %}
          <p class="preview-card__desc">
            <a href="{% url 'datasets' %}" style="color:var(--color-accent);font-weight:600;">
              {{ stats.active_connections }} node{{ stats.active_connections|pluralize }} connected
            </a>
          </p>
        {% else %}
          <p class="preview-card__desc">Connected nodes and their health indicators</p>
          <span class="preview-card__badge">No connections</span>
        {% endif %}
      </div>
    </div>

  </div><!-- /Right column stack -->

</div><!-- /Layout Y -->
```

- [ ] **Step 5c.2: Remove the old `switchTab` script at the bottom of the file**

The tab-switching script (lines 377–393) is no longer needed. Delete it:

```html
<!-- DELETE THIS ENTIRE BLOCK:
<script>
function switchTab(tab) { ... }
</script>
-->
```

- [ ] **Step 5c.3: Keep the Pro Tip script**

The Pro Tip `<script>` block (lines 194–251) must be kept. Do not remove it.

- [ ] **Step 5d: Smoke test**

Open `http://localhost:8000/panel/`. Verify:
1. No "Dashboard" H1 — only the topbar shows "Dashboard"
2. Setup banner appears (step 1 if no connections)
3. Layout Y renders — Active Trainings on left, 3 stacked cards on right
4. Ghost chart shows in Active Trainings when no jobs
5. Connections stat card shows amber left border + "Add a connection" link when count = 0
6. Pro Tip still renders below the featured card

- [ ] **Step 5e: Commit**

```bash
git add MediNetHub/templates/webapp/dashboard_home.html
git commit -m "feat: redesign dashboard empty state with banner B and layout Y"
```

---

## Task 6: Final verification

- [ ] **Step 6.1: Check banner step progression manually**

Using Django shell or admin, create test data to verify each step:

```bash
python manage.py shell
```

```python
from webapp.models import Connection, Dataset, ModelConfig, TrainingJob
from django.contrib.auth.models import User

u = User.objects.get(username='admin')

# Step 1: no connections → banner shows step 1
print(Connection.objects.filter(user=u).count())  # should be 0
```

Visit `/panel/` — banner says "start by adding your first data connection", dot 1 is blue.

- [ ] **Step 6.2: Verify sidebar lock**

With no connections, all three sidebar items (Datasets, Model Studio, Training) should be faded and non-clickable. Dashboard and Notifications remain active.

- [ ] **Step 6.3: Verify banner disappears when jobs exist**

If an active job exists in the DB, the banner should not render. Verify by checking a user account that has running/pending jobs.

- [ ] **Step 6.4: Final commit**

```bash
git add -A
git commit -m "feat: dashboard empty state redesign complete"
```
