"""
hitl_review.py
--------------
Human-in-the-Loop (HITL) review interface for LLM-extracted metadata.
Launches a local web server where a human can review, correct, and approve
each metadata record before it is written to approved_metadata.json.

Tracks all human corrections in audit_trail.json for full transparency.

Usage:
    python src/hitl_review.py --input results/metadata.json
    python src/hitl_review.py --input results/metadata.json --port 8080
"""

import json
import argparse
import os
import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import webbrowser
import threading

# ── Paths ─────────────────────────────────────────────────────────────────────

APPROVED_PATH  = "results/approved_metadata.json"
AUDIT_PATH     = "results/audit_trail.json"

# ── State ─────────────────────────────────────────────────────────────────────

state = {
    "items":    [],
    "approved": [],
    "audit":    [],
    "current":  0,
    "done":     False,
}

# ── HTML template ─────────────────────────────────────────────────────────────

def render_page(item: dict, index: int, total: int) -> str:
    """Render the review form for a single metadata item."""

    corrections_html = ""
    if item.get("llm_corrections"):
        rows = ""
        for field, change in item["llm_corrections"].items():
            before = change.get("before", "")
            after  = change.get("after", "")
            rows += f"""
            <tr>
                <td class="field-name">{field}</td>
                <td class="before">{before}</td>
                <td class="after">{after}</td>
            </tr>"""
        corrections_html = f"""
        <div class="section">
            <h3>⚡ LLM Corrections Made</h3>
            <table class="corrections-table">
                <thead><tr><th>Field</th><th>Before (ML)</th><th>After (LLM)</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>"""

    certs = ", ".join(item.get("certifications") or []) if item.get("certifications") else ""
    ml_conf_pct = f"{item.get('ml_confidence', 0) * 100:.1f}%"

    fields = [
        ("component_type",  "Component Type",   item.get("component_type", ""),   "text"),
        ("material",        "Material",          item.get("material", ""),         "text"),
        ("voltage_v",       "Voltage (V)",       item.get("voltage_v", ""),        "number"),
        ("capacity_ah",     "Capacity (Ah)",     item.get("capacity_ah", ""),      "number"),
        ("weight_kg",       "Weight (kg)",       item.get("weight_kg", ""),        "number"),
        ("temperature_range","Temperature Range",item.get("temperature_range",""), "text"),
        ("certifications",  "Certifications",    certs,                            "text"),
        ("manufacturer",    "Manufacturer",      item.get("manufacturer", ""),     "text"),
        ("part_number",     "Part Number",       item.get("part_number", ""),      "text"),
    ]

    fields_html = ""
    for key, label, value, ftype in fields:
        val = value if value is not None else ""
        fields_html += f"""
        <div class="field-row">
            <label for="{key}">{label}</label>
            <input type="{ftype}" id="{key}" name="{key}" value="{val}" step="any">
        </div>"""

    progress_pct = int((index / total) * 100)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HITL Review — {item['item_id']}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
            background: #0d1117;
            color: #c9d1d9;
            min-height: 100vh;
            padding: 0;
        }}

        .topbar {{
            background: #161b22;
            border-bottom: 1px solid #21262d;
            padding: 14px 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .topbar-title {{
            font-size: 13px;
            color: #58a6ff;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}

        .topbar-meta {{
            font-size: 12px;
            color: #6e7681;
        }}

        .progress-bar-wrap {{
            background: #21262d;
            height: 3px;
            width: 100%;
        }}

        .progress-bar-fill {{
            background: #238636;
            height: 3px;
            width: {progress_pct}%;
            transition: width 0.3s ease;
        }}

        .container {{
            max-width: 860px;
            margin: 36px auto;
            padding: 0 24px;
        }}

        .item-header {{
            margin-bottom: 28px;
        }}

        .item-id {{
            font-size: 22px;
            font-weight: 700;
            color: #f0f6fc;
            margin-bottom: 6px;
        }}

        .item-badges {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 10px;
        }}

        .badge {{
            font-size: 11px;
            padding: 3px 10px;
            border-radius: 20px;
            font-weight: 600;
            letter-spacing: 0.04em;
        }}

        .badge-ml {{
            background: #1f2937;
            color: #9ca3af;
            border: 1px solid #374151;
        }}

        .badge-llm {{
            background: #0d2137;
            color: #58a6ff;
            border: 1px solid #1d4ed8;
        }}

        .badge-conf {{
            background: #1a1a2e;
            color: #a78bfa;
            border: 1px solid #6d28d9;
        }}

        .section {{
            background: #161b22;
            border: 1px solid #21262d;
            border-radius: 8px;
            padding: 20px 24px;
            margin-bottom: 20px;
        }}

        .section h3 {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #6e7681;
            margin-bottom: 16px;
        }}

        .corrections-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }}

        .corrections-table th {{
            text-align: left;
            color: #6e7681;
            font-weight: 600;
            padding: 6px 10px;
            border-bottom: 1px solid #21262d;
        }}

        .corrections-table td {{
            padding: 7px 10px;
            border-bottom: 1px solid #161b22;
        }}

        .field-name {{ color: #79c0ff; }}
        .before {{ color: #f85149; text-decoration: line-through; opacity: 0.7; }}
        .after {{ color: #56d364; }}

        .field-row {{
            display: grid;
            grid-template-columns: 180px 1fr;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }}

        label {{
            font-size: 12px;
            color: #8b949e;
            text-align: right;
        }}

        input[type="text"],
        input[type="number"] {{
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #f0f6fc;
            font-family: inherit;
            font-size: 13px;
            padding: 7px 12px;
            width: 100%;
            transition: border-color 0.2s;
        }}

        input:focus {{
            outline: none;
            border-color: #58a6ff;
        }}

        .notes-row {{
            margin-top: 8px;
        }}

        textarea {{
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #f0f6fc;
            font-family: inherit;
            font-size: 13px;
            padding: 10px 12px;
            width: 100%;
            resize: vertical;
            min-height: 72px;
            transition: border-color 0.2s;
        }}

        textarea:focus {{
            outline: none;
            border-color: #58a6ff;
        }}

        .actions {{
            display: flex;
            gap: 12px;
            margin-top: 8px;
        }}

        .btn {{
            flex: 1;
            padding: 12px 20px;
            border: none;
            border-radius: 6px;
            font-family: inherit;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            letter-spacing: 0.03em;
            transition: opacity 0.15s;
        }}

        .btn:hover {{ opacity: 0.85; }}

        .btn-approve {{
            background: #238636;
            color: #f0f6fc;
        }}

        .btn-reject {{
            background: #da3633;
            color: #f0f6fc;
            flex: 0 0 auto;
            padding: 12px 24px;
        }}

        .counter {{
            font-size: 12px;
            color: #6e7681;
            text-align: center;
            margin-top: 16px;
        }}
    </style>
</head>
<body>

<div class="topbar">
    <span class="topbar-title">⚙ HITL Metadata Review</span>
    <span class="topbar-meta">Item {index + 1} of {total}</span>
</div>
<div class="progress-bar-wrap">
    <div class="progress-bar-fill"></div>
</div>

<div class="container">

    <div class="item-header">
        <div class="item-id">{item['item_id']}</div>
        <div class="item-badges">
            <span class="badge badge-ml">ML: {item.get('ml_label','?')}</span>
            <span class="badge badge-conf">ML conf: {ml_conf_pct}</span>
            <span class="badge badge-llm">LLM validated ✓</span>
        </div>
    </div>

    {corrections_html}

    <form method="POST" action="/submit">
        <input type="hidden" name="item_id" value="{item['item_id']}">

        <div class="section">
            <h3>📋 Review & Correct Fields</h3>
            {fields_html}
            <div class="notes-row field-row">
                <label for="human_notes">Your Notes</label>
                <textarea id="human_notes" name="human_notes" placeholder="Optional: note any concerns or corrections made..."></textarea>
            </div>
        </div>

        <div class="actions">
            <button type="submit" name="action" value="approve" class="btn btn-approve">
                ✅ Approve & Continue
            </button>
            <button type="submit" name="action" value="reject" class="btn btn-reject">
                ❌ Reject
            </button>
        </div>
    </form>

    <div class="counter">
        {len(state['approved'])} approved so far · {total - index - 1} remaining
    </div>

</div>
</body>
</html>"""


def render_done_page(approved: int, rejected: int, audit: list) -> str:
    """Render the completion summary page."""
    rows = ""
    for entry in audit:
        changes = len(entry.get("human_corrections", {}))
        status_color = "#56d364" if entry["status"] == "approved" else "#f85149"
        rows += f"""
        <tr>
            <td>{entry['item_id']}</td>
            <td style="color:{status_color}">{entry['status'].upper()}</td>
            <td>{changes} field(s) corrected</td>
            <td style="color:#6e7681;font-size:11px">{entry['reviewed_at']}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HITL Review — Complete</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'SF Mono', 'Fira Code', monospace;
            background: #0d1117;
            color: #c9d1d9;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }}
        .card {{
            background: #161b22;
            border: 1px solid #21262d;
            border-radius: 12px;
            padding: 40px 48px;
            max-width: 680px;
            width: 100%;
            text-align: center;
        }}
        .icon {{ font-size: 48px; margin-bottom: 16px; }}
        h1 {{ font-size: 22px; color: #f0f6fc; margin-bottom: 8px; }}
        .sub {{ font-size: 13px; color: #6e7681; margin-bottom: 28px; }}
        .stats {{
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-bottom: 28px;
        }}
        .stat {{
            background: #0d1117;
            border: 1px solid #21262d;
            border-radius: 8px;
            padding: 14px 24px;
        }}
        .stat-num {{ font-size: 28px; font-weight: 700; color: #58a6ff; }}
        .stat-label {{ font-size: 11px; color: #6e7681; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 12px; }}
        th {{ color: #6e7681; padding: 8px; border-bottom: 1px solid #21262d; }}
        td {{ padding: 8px; border-bottom: 1px solid #161b22; color: #c9d1d9; }}
        .paths {{ margin-top: 24px; font-size: 11px; color: #6e7681; line-height: 1.8; }}
        .path {{ color: #58a6ff; }}
    </style>
</head>
<body>
<div class="card">
    <div class="icon">✅</div>
    <h1>Review Complete</h1>
    <p class="sub">All items have been reviewed. Results saved.</p>

    <div class="stats">
        <div class="stat">
            <div class="stat-num" style="color:#56d364">{approved}</div>
            <div class="stat-label">Approved</div>
        </div>
        <div class="stat">
            <div class="stat-num" style="color:#f85149">{rejected}</div>
            <div class="stat-label">Rejected</div>
        </div>
        <div class="stat">
            <div class="stat-num">{approved + rejected}</div>
            <div class="stat-label">Total</div>
        </div>
    </div>

    <table>
        <thead><tr><th>Item</th><th>Status</th><th>Corrections</th><th>Reviewed At</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>

    <div class="paths">
        Approved data → <span class="path">{APPROVED_PATH}</span><br>
        Audit trail &nbsp;→ <span class="path">{AUDIT_PATH}</span>
    </div>
</div>
</body>
</html>"""


# ── HTTP Handler ──────────────────────────────────────────────────────────────

class HITLHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # suppress default HTTP logs

    def do_GET(self):
        if state["done"]:
            approved = sum(1 for a in state["audit"] if a["status"] == "approved")
            rejected = sum(1 for a in state["audit"] if a["status"] == "rejected")
            html = render_done_page(approved, rejected, state["audit"])
        elif state["current"] < len(state["items"]):
            item  = state["items"][state["current"]]
            html  = render_page(item, state["current"], len(state["items"]))
        else:
            html = "<h1>No items to review.</h1>"

        self._send_html(html)

    def do_POST(self):
        length  = int(self.headers.get("Content-Length", 0))
        body    = self.rfile.read(length).decode("utf-8")
        params  = parse_qs(body)

        def get(key):
            vals = params.get(key, [""])
            return vals[0].strip()

        action   = get("action")
        item_id  = get("item_id")
        notes    = get("human_notes")

        # Find original item
        original = next((i for i in state["items"] if i["item_id"] == item_id), {})

        # Build corrected item
        corrected = dict(original)

        def coerce(key, val, original_val):
            if val == "":
                return None
            if isinstance(original_val, float) or isinstance(original_val, int):
                try: return float(val)
                except: return val
            return val

        editable_fields = [
            "component_type", "material", "voltage_v", "capacity_ah",
            "weight_kg", "temperature_range", "manufacturer", "part_number"
        ]

        human_corrections = {}
        for field in editable_fields:
            new_val = get(field)
            orig_val = original.get(field)
            coerced = coerce(field, new_val, orig_val)
            corrected[field] = coerced
            if str(coerced) != str(orig_val):
                human_corrections[field] = {
                    "llm_value": orig_val,
                    "human_value": coerced
                }

        # Certifications — comma separated
        certs_raw = get("certifications")
        cert_list = [c.strip() for c in certs_raw.split(",") if c.strip()]
        if cert_list != original.get("certifications", []):
            human_corrections["certifications"] = {
                "llm_value": original.get("certifications", []),
                "human_value": cert_list
            }
        corrected["certifications"] = cert_list

        # Add HITL metadata
        corrected["hitl_status"]      = action
        corrected["hitl_reviewed_at"] = datetime.datetime.now().isoformat()
        corrected["hitl_notes"]       = notes
        corrected["human_corrections"] = human_corrections

        # Audit entry
        audit_entry = {
            "item_id":           item_id,
            "status":            action,
            "reviewed_at":       corrected["hitl_reviewed_at"],
            "human_corrections": human_corrections,
            "human_notes":       notes,
            "ml_confidence":     original.get("ml_confidence"),
            "llm_corrections_count": len(original.get("llm_corrections", {})),
            "human_corrections_count": len(human_corrections),
        }

        state["audit"].append(audit_entry)

        if action == "approve":
            state["approved"].append(corrected)

        state["current"] += 1

        # Check if all done
        if state["current"] >= len(state["items"]):
            state["done"] = True
            _save_results()

        # Redirect back to GET
        self.send_response(302)
        self.send_header("Location", "/")
        self.end_headers()

    def _send_html(self, html: str):
        encoded = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


# ── Save results ──────────────────────────────────────────────────────────────

def _save_results():
    os.makedirs("results", exist_ok=True)

    with open(APPROVED_PATH, "w") as f:
        json.dump(state["approved"], f, indent=2)

    with open(AUDIT_PATH, "w") as f:
        json.dump({
            "reviewed_at": datetime.datetime.now().isoformat(),
            "total_items": len(state["items"]),
            "approved":    len(state["approved"]),
            "rejected":    len(state["items"]) - len(state["approved"]),
            "entries":     state["audit"],
        }, f, indent=2)

    print(f"\n✅ Review complete.")
    print(f"   Approved metadata → {APPROVED_PATH}")
    print(f"   Audit trail       → {AUDIT_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="HITL review interface for LLM metadata")
    p.add_argument("--input", default="results/metadata.json", help="Path to metadata.json")
    p.add_argument("--port",  type=int, default=5050,          help="Local port (default: 5050)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        return

    with open(args.input) as f:
        state["items"] = json.load(f)

    print(f"\n{'='*55}")
    print(f"  HITL Metadata Review")
    print(f"{'='*55}")
    print(f"  Input:  {args.input}")
    print(f"  Items:  {len(state['items'])} to review")
    print(f"  URL:    http://localhost:{args.port}")
    print(f"{'='*55}\n")

    url = f"http://localhost:{args.port}"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    server = HTTPServer(("localhost", args.port), HITLHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        if not state["done"]:
            _save_results()


if __name__ == "__main__":
    main()
