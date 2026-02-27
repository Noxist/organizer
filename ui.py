"""
ui.py – HTML templates for the Organizer web UI.

Design: clean, neutral, functional. Visually consistent with HomeBox.
No external CDN dependencies – all CSS inline.
"""

# ---------------------------------------------------------------------------
# Base layout
# ---------------------------------------------------------------------------

def _base(title: str, content: str, nav_active: str = "", homebox_url: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} – Organizer</title>
<style>
:root {{
  --bg: #f8f9fa;
  --card: #fff;
  --primary: #4a6cf7;
  --primary-hover: #3b5de7;
  --danger: #ef4444;
  --success: #22c55e;
  --warning: #f59e0b;
  --text: #1f2937;
  --text-muted: #6b7280;
  --border: #e5e7eb;
  --radius: 8px;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); }}
a {{ color: var(--primary); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

/* Nav */
nav {{ background: var(--card); border-bottom: 1px solid var(--border); padding: 0 1.5rem; display: flex; align-items: center; gap: 2rem; height: 56px; }}
nav .logo {{ font-weight: 700; font-size: 1.1rem; color: var(--primary); }}
nav a {{ color: var(--text-muted); font-size: 0.9rem; padding: 0.5rem 0; border-bottom: 2px solid transparent; }}
nav a:hover, nav a.active {{ color: var(--primary); border-bottom-color: var(--primary); text-decoration: none; }}

/* Layout */
.container {{ max-width: 1000px; margin: 0 auto; padding: 1.5rem; }}
.grid {{ display: grid; gap: 1rem; }}
.grid-2 {{ grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }}
.grid-3 {{ grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }}

/* Cards */
.card {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.25rem; }}
.card h3 {{ margin-bottom: 0.75rem; font-size: 1rem; }}
.card .meta {{ color: var(--text-muted); font-size: 0.85rem; }}

/* Buttons */
.btn {{ display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.5rem 1rem; border: none; border-radius: var(--radius); font-size: 0.875rem; font-weight: 500; cursor: pointer; transition: background 0.15s; }}
.btn-primary {{ background: var(--primary); color: #fff; }}
.btn-primary:hover {{ background: var(--primary-hover); }}
.btn-danger {{ background: var(--danger); color: #fff; }}
.btn-sm {{ padding: 0.3rem 0.6rem; font-size: 0.8rem; }}
.btn-outline {{ background: transparent; border: 1px solid var(--border); color: var(--text); }}
.btn-outline:hover {{ background: var(--bg); }}

/* Forms */
.form-group {{ margin-bottom: 1rem; }}
.form-group label {{ display: block; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.3rem; color: var(--text-muted); }}
.form-group input, .form-group select, .form-group textarea {{ width: 100%; padding: 0.5rem 0.75rem; border: 1px solid var(--border); border-radius: var(--radius); font-size: 0.9rem; }}
.form-group input:focus, .form-group select:focus {{ outline: none; border-color: var(--primary); box-shadow: 0 0 0 2px rgba(74,108,247,0.15); }}
.form-row {{ display: flex; gap: 0.75rem; }}
.form-row > * {{ flex: 1; }}

/* Tags */
.tag {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 999px; font-size: 0.75rem; font-weight: 500; }}
.tag-blue {{ background: #dbeafe; color: #1e40af; }}
.tag-green {{ background: #dcfce7; color: #166534; }}
.tag-yellow {{ background: #fef3c7; color: #92400e; }}
.tag-red {{ background: #fee2e2; color: #991b1b; }}
.tag-gray {{ background: #f3f4f6; color: #374151; }}

/* Slot grid */
.slot-grid {{ display: grid; gap: 2px; }}
.slot {{ width: 100%; aspect-ratio: 1; border: 1px solid var(--border); border-radius: 3px; display: flex; align-items: center; justify-content: center; font-size: 0.65rem; color: var(--text-muted); cursor: default; }}
.slot.occupied {{ background: var(--primary); color: #fff; border-color: var(--primary); }}
.slot.empty {{ background: #f9fafb; }}

/* Stats */
.stat {{ text-align: center; }}
.stat .num {{ font-size: 2rem; font-weight: 700; color: var(--primary); }}
.stat .label {{ font-size: 0.8rem; color: var(--text-muted); }}

/* Upload zone */
.upload-zone {{ border: 2px dashed var(--border); border-radius: var(--radius); padding: 2rem; text-align: center; color: var(--text-muted); cursor: pointer; transition: border-color 0.2s; }}
.upload-zone:hover {{ border-color: var(--primary); }}
.upload-zone.dragover {{ border-color: var(--primary); background: rgba(74,108,247,0.05); }}

/* Toast */
.toast {{ position: fixed; bottom: 1.5rem; right: 1.5rem; padding: 0.75rem 1.25rem; border-radius: var(--radius); color: #fff; font-size: 0.9rem; z-index: 1000; animation: fadeIn 0.3s; }}
.toast-success {{ background: var(--success); }}
.toast-error {{ background: var(--danger); }}
@keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}

/* Alert */
.alert {{ padding: 0.75rem 1rem; border-radius: var(--radius); font-size: 0.875rem; margin-bottom: 1rem; }}
.alert-info {{ background: #dbeafe; color: #1e40af; }}
.alert-warn {{ background: #fef3c7; color: #92400e; }}
.alert-error {{ background: #fee2e2; color: #991b1b; }}

/* Misc */
.mb-1 {{ margin-bottom: 1rem; }}
.mt-1 {{ margin-top: 1rem; }}
.text-muted {{ color: var(--text-muted); }}
.text-sm {{ font-size: 0.85rem; }}
hr {{ border: none; border-top: 1px solid var(--border); margin: 1rem 0; }}
</style>
</head>
<body>
<nav>
  <span class="logo">📦 Organizer</span>
  <a href="/" class="{'active' if nav_active == 'dashboard' else ''}">Dashboard</a>
  <a href="/scan" class="{'active' if nav_active == 'scan' else ''}">Scan</a>
  <a href="/containers" class="{'active' if nav_active == 'containers' else ''}">Container</a>
  <a href="/items" class="{'active' if nav_active == 'items' else ''}">Items</a>
  {f'<a href="{homebox_url}" target="_blank" style="margin-left:auto;">🏠 HomeBox</a>' if homebox_url else ''}
</nav>
<div class="container">
{content}
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def dashboard_html(stats: dict, homebox_ok: bool, homebox_url: str = "") -> str:
    hb_status = '<span class="tag tag-green">Online</span>' if homebox_ok else '<span class="tag tag-red">Offline</span>'
    content = f"""
<h2 style="margin-bottom:1rem;">Dashboard</h2>
<div class="grid grid-3 mb-1">
  <div class="card stat"><div class="num">{stats.get('total_items', 0)}</div><div class="label">Items total</div></div>
  <div class="card stat"><div class="num">{stats.get('placed_items', 0)}</div><div class="label">Platziert</div></div>
  <div class="card stat"><div class="num">{stats.get('unplaced_items', 0)}</div><div class="label">Unplatziert</div></div>
</div>
<div class="grid grid-3 mb-1">
  <div class="card stat"><div class="num">{stats.get('total_containers', 0)}</div><div class="label">Container</div></div>
  <div class="card stat"><div class="num">{stats.get('free_slots', 0)}</div><div class="label">Freie Slots</div></div>
  <div class="card stat"><div class="num">{stats.get('utilization_pct', 0)}%</div><div class="label">Auslastung</div></div>
</div>
<div class="card mb-1">
  <h3>System</h3>
  <p class="text-sm">HomeBox: {hb_status}</p>
</div>
<div class="grid grid-2">
  <a href="/scan" class="card" style="text-align:center;padding:2rem;">
    <div style="font-size:2rem;margin-bottom:0.5rem;">📷</div>
    <strong>Neues Item scannen</strong>
  </a>
  <a href="/containers" class="card" style="text-align:center;padding:2rem;">
    <div style="font-size:2rem;margin-bottom:0.5rem;">📦</div>
    <strong>Container verwalten</strong>
  </a>
</div>
"""
    return _base("Dashboard", content, "dashboard", homebox_url)


# ---------------------------------------------------------------------------
# Scan page
# ---------------------------------------------------------------------------

def scan_html(openai_enabled: bool = False, homebox_url: str = "") -> str:
    ai_note = ""
    if openai_enabled:
        ai_note = '<p class="text-sm text-muted mb-1">✨ KI-Erkennung aktiv — Name und Tags werden vorgeschlagen.</p>'
    else:
        ai_note = '<p class="text-sm text-muted mb-1">KI-Erkennung deaktiviert — Name manuell eingeben.</p>'

    content = f"""
<h2 style="margin-bottom:1rem;">Item scannen</h2>
{ai_note}
<div class="card mb-1">
  <h3>1. Foto aufnehmen</h3>
  <p class="text-sm text-muted mb-1">Objekt auf 2cm-Raster legen, top-down fotografieren.</p>
  <div id="upload-zone" class="upload-zone" onclick="document.getElementById('file-input').click()">
    <input type="file" id="file-input" accept="image/*" capture="environment" style="display:none" onchange="handleUpload(this)">
    <p>📷 Foto aufnehmen oder Bild hochladen</p>
    <p class="text-sm">Klicken oder Datei hierher ziehen</p>
  </div>
  <div id="preview" style="margin-top:1rem;display:none;">
    <img id="preview-img" style="max-width:100%;border-radius:var(--radius);border:1px solid var(--border);">
  </div>
</div>

<div id="measure-result" style="display:none;">
<div class="card mb-1">
  <h3>2. Messergebnis</h3>
  <div class="form-row">
    <div class="form-group">
      <label>Breite (mm)</label>
      <input type="number" id="width_mm" readonly>
    </div>
    <div class="form-group">
      <label>Tiefe (mm)</label>
      <input type="number" id="depth_mm" readonly>
    </div>
    <div class="form-group">
      <label>Höhe (mm, optional)</label>
      <input type="number" id="height_mm" placeholder="manuell">
    </div>
  </div>
  <div id="debug-img-container" style="display:none;margin-top:0.5rem;">
    <img id="debug-img" style="max-width:100%;border-radius:var(--radius);">
  </div>
</div>

<div class="card mb-1">
  <h3>3. Item Details</h3>
  <div id="ai-suggestion" style="display:none;" class="alert alert-info mb-1">
    <strong>KI-Vorschlag:</strong> <span id="ai-name"></span>
    <br><span class="text-sm" id="ai-tags"></span>
  </div>
  <div class="form-group">
    <label>Name *</label>
    <input type="text" id="item-name" placeholder="z.B. USB Hub" required>
  </div>
  <div class="form-group">
    <label>Tags (kommagetrennt)</label>
    <input type="text" id="item-tags" placeholder="z.B. Elektronik, USB, Büro">
  </div>
  <div class="form-group">
    <label>Beschreibung</label>
    <textarea id="item-desc" rows="2" placeholder="Optional"></textarea>
  </div>
</div>

<div class="card">
  <h3>4. Bestätigen</h3>
  <p class="text-sm text-muted mb-1">Item wird in HomeBox angelegt und einem Slot zugewiesen.</p>
  <button class="btn btn-primary" onclick="confirmItem()" id="confirm-btn">
    ✓ Item anlegen &amp; zuweisen
  </button>
  <span id="confirm-status" class="text-sm" style="margin-left:1rem;"></span>
</div>
</div>

<script>
let scanData = {{}};

function handleUpload(input) {{
  const file = input.files[0];
  if (!file) return;

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {{
    document.getElementById('preview').style.display = 'block';
    document.getElementById('preview-img').src = e.target.result;
  }};
  reader.readAsDataURL(file);

  // Upload for measurement
  const formData = new FormData();
  formData.append('image', file);

  document.getElementById('upload-zone').innerHTML = '<p>⏳ Wird gemessen...</p>';

  fetch('/api/scan/measure', {{ method: 'POST', body: formData }})
    .then(r => r.json())
    .then(data => {{
      if (data.error) {{
        document.getElementById('upload-zone').innerHTML = '<p style="color:var(--danger);">❌ ' + data.error + '</p><p class="text-sm">Erneut versuchen: klicken</p>';
        document.getElementById('upload-zone').onclick = () => document.getElementById('file-input').click();
        return;
      }}
      scanData = data;
      document.getElementById('width_mm').value = Math.round(data.width_mm);
      document.getElementById('depth_mm').value = Math.round(data.depth_mm);
      document.getElementById('measure-result').style.display = 'block';

      if (data.debug_image) {{
        document.getElementById('debug-img').src = 'data:image/jpeg;base64,' + data.debug_image;
        document.getElementById('debug-img-container').style.display = 'block';
      }}

      // AI identification (if available)
      if (data.ai_result) {{
        document.getElementById('ai-suggestion').style.display = 'block';
        document.getElementById('ai-name').textContent = data.ai_result.name;
        document.getElementById('ai-tags').textContent = 'Tags: ' + data.ai_result.tags.join(', ');
        document.getElementById('item-name').value = data.ai_result.name;
        document.getElementById('item-tags').value = data.ai_result.tags.join(', ');
        if (data.ai_result.description) {{
          document.getElementById('item-desc').value = data.ai_result.description;
        }}
      }}

      document.getElementById('upload-zone').innerHTML = '<p style="color:var(--success);">✓ Messung erfolgreich</p>';
    }})
    .catch(err => {{
      document.getElementById('upload-zone').innerHTML = '<p style="color:var(--danger);">❌ Fehler: ' + err.message + '</p>';
    }});
}}

function confirmItem() {{
  const name = document.getElementById('item-name').value.trim();
  if (!name) {{ alert('Name ist erforderlich.'); return; }}

  const btn = document.getElementById('confirm-btn');
  btn.disabled = true;
  btn.textContent = '⏳ Wird angelegt...';

  const tags = document.getElementById('item-tags').value.split(',').map(t => t.trim()).filter(Boolean);
  const height = parseInt(document.getElementById('height_mm').value) || null;

  fetch('/api/items', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{
      name: name,
      tags: tags,
      description: document.getElementById('item-desc').value.trim(),
      width_mm: parseInt(document.getElementById('width_mm').value),
      depth_mm: parseInt(document.getElementById('depth_mm').value),
      height_mm: height,
      scan_hash: scanData.scan_hash || '',
    }})
  }})
  .then(r => r.json())
  .then(data => {{
    if (data.error) {{
      document.getElementById('confirm-status').innerHTML = '<span style="color:var(--danger);">❌ ' + data.error + '</span>';
      btn.disabled = false;
      btn.textContent = '✓ Item anlegen & zuweisen';
      return;
    }}
    document.getElementById('confirm-status').innerHTML =
      '<span style="color:var(--success);">✓ ' + data.name + ' → ' + data.address + '</span>';
    btn.textContent = '✓ Fertig';
  }})
  .catch(err => {{
    document.getElementById('confirm-status').innerHTML = '<span style="color:var(--danger);">❌ ' + err.message + '</span>';
    btn.disabled = false;
    btn.textContent = '✓ Item anlegen & zuweisen';
  }});
}}

// Drag & drop
const zone = document.getElementById('upload-zone');
zone.addEventListener('dragover', (e) => {{ e.preventDefault(); zone.classList.add('dragover'); }});
zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
zone.addEventListener('drop', (e) => {{
  e.preventDefault();
  zone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) {{
    const input = document.getElementById('file-input');
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    handleUpload(input);
  }}
}});
</script>
"""
    return _base("Scan", content, "scan", homebox_url)


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

def containers_html(containers: dict, slots: dict, homebox_url: str = "") -> str:
    cards = ""
    for cid, c in sorted(containers.items()):
        c_slots = {k: v for k, v in slots.items() if v["container_id"] == cid}
        total = len(c_slots)
        occupied = sum(1 for s in c_slots.values() if s["occupied"])
        pct = round(occupied / total * 100) if total > 0 else 0
        lock_icon = "🔒" if c.get("locked") else ""

        cols = c.get("cols", 1)
        slot_cells = ""
        for sk in sorted(c_slots.keys()):
            sv = c_slots[sk]
            cls = "occupied" if sv["occupied"] else "empty"
            label = sk.split("/")[-1]
            title = sv.get("item_id", "") if sv["occupied"] else "frei"
            slot_cells += f'<div class="slot {cls}" title="{title}">{label}</div>'

        cards += f"""
<div class="card mb-1">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <h3>{cid} — {c['name']} {lock_icon}</h3>
    <span class="tag tag-{'green' if pct < 80 else 'yellow' if pct < 100 else 'red'}">{occupied}/{total} ({pct}%)</span>
  </div>
  <p class="text-sm text-muted">📍 {c['location']} · {c['width_mm']}×{c['depth_mm']}×{c['height_mm']}mm</p>
  <div class="slot-grid mt-1" style="grid-template-columns: repeat({cols}, 1fr);">
    {slot_cells}
  </div>
  <div style="margin-top:0.75rem;">
    <form method="POST" action="/api/containers/{cid}/delete" style="display:inline;">
      <button class="btn btn-danger btn-sm" onclick="return confirm('Container {cid} wirklich löschen?')">Löschen</button>
    </form>
  </div>
</div>"""

    content = f"""
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
  <h2>Container</h2>
</div>

{cards if cards else '<p class="text-muted">Noch keine Container angelegt.</p>'}

<div class="card mt-1">
  <h3>Neuer Container</h3>
  <form method="POST" action="/api/containers">
    <div class="form-row">
      <div class="form-group">
        <label>ID *</label>
        <input type="text" name="id" placeholder="z.B. R-01" required pattern="[A-Za-z0-9_-]+">
      </div>
      <div class="form-group">
        <label>Name *</label>
        <input type="text" name="name" placeholder="z.B. IKEA KALLAX Fach 1" required>
      </div>
    </div>
    <div class="form-group">
      <label>Standort *</label>
      <input type="text" name="location" placeholder="z.B. Wohnzimmer" required>
    </div>
    <div class="form-row">
      <div class="form-group">
        <label>Breite (mm) *</label>
        <input type="number" name="width_mm" placeholder="330" required min="1">
      </div>
      <div class="form-group">
        <label>Tiefe (mm) *</label>
        <input type="number" name="depth_mm" placeholder="330" required min="1">
      </div>
      <div class="form-group">
        <label>Höhe (mm) *</label>
        <input type="number" name="height_mm" placeholder="390" required min="1">
      </div>
    </div>
    <button type="submit" class="btn btn-primary">Container anlegen</button>
  </form>
</div>
"""
    return _base("Container", content, "containers", homebox_url)


# ---------------------------------------------------------------------------
# Items list
# ---------------------------------------------------------------------------

def items_html(items: dict, slots: dict, containers: dict, homebox_url: str = "") -> str:
    rows = ""
    for iid, item in sorted(items.items(), key=lambda x: x[1].get("created_at", ""), reverse=True):
        slot_id = item.get("slot_id", "UNPLACED")
        if slot_id == "UNPLACED":
            address = '<span class="tag tag-yellow">UNPLACED</span>'
        else:
            slot = slots.get(slot_id, {})
            cid = slot.get("container_id", "?")
            cont = containers.get(cid, {})
            loc = cont.get("location", "?")
            slot_label = slot_id.split("/")[-1]
            address = f'<span class="tag tag-green">{loc} / {cid} / {slot_label}</span>'

        tags_html = " ".join(f'<span class="tag tag-blue">{t}</span>' for t in item.get("tags", []))
        dims = f"{item.get('width_mm', '?')}×{item.get('depth_mm', '?')}"
        if item.get("height_mm"):
            dims += f"×{item['height_mm']}"
        dims += "mm"

        rows += f"""
<div class="card mb-1" style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">
  <div>
    <strong>{item['name']}</strong>
    <span class="text-sm text-muted" style="margin-left:0.5rem;">{dims}</span>
    <div style="margin-top:0.3rem;">{tags_html}</div>
  </div>
  <div style="text-align:right;">
    {address}
    <div style="margin-top:0.3rem;">
      <form method="POST" action="/api/items/{iid}/remove" style="display:inline;">
        <button class="btn btn-outline btn-sm">Entfernen</button>
      </form>
    </div>
  </div>
</div>"""

    unplaced_count = sum(1 for i in items.values() if i.get("slot_id") == "UNPLACED")
    pack_section = ""
    if unplaced_count > 0:
        pack_section = f"""
<div class="alert alert-warn mb-1">
  {unplaced_count} Item(s) nicht zugewiesen.
  <form method="POST" action="/api/pack" style="display:inline;margin-left:1rem;">
    <button class="btn btn-primary btn-sm">Auto-Pack ausführen</button>
  </form>
</div>"""

    content = f"""
<h2 style="margin-bottom:1rem;">Items ({len(items)})</h2>
{pack_section}
{rows if rows else '<p class="text-muted">Noch keine Items registriert.</p>'}
"""
    return _base("Items", content, "items", homebox_url)
