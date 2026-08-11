"""
Lightweight web dashboard to toggle employee door access.
Reads/writes employees_auth.csv so recognize.py picks up changes live.
Run automatically by recognize.py, or standalone: python dashboard.py
"""

import csv
import json
import os
from functools import wraps

from flask import Flask, render_template_string, request, jsonify

import config

app = Flask(__name__)

# ------------------------------------------------------------------ basic auth
def check_auth(username, password):
    return username == config.DASHBOARD_USERNAME and password == config.DASHBOARD_PASSWORD

def authenticate():
    return ("Unauthorized", 401, {"WWW-Authenticate": 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# ------------------------------------------------------------------ CSV utils
def load_auth_csv():
    auth = {}
    if not os.path.exists(config.AUTH_CSV_FILE):
        return auth
    with open(config.AUTH_CSV_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "name" in row and "authorized" in row:
                auth[row["name"].strip()] = row["authorized"].strip().lower() in ("true", "1", "yes", "on")
    return auth

def save_auth_csv(auth_dict):
    with open(config.AUTH_CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "authorized"])
        for name, authorized in sorted(auth_dict.items()):
            writer.writerow([name, "true" if authorized else "false"])

# ------------------------------------------------------------------ templates
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Door Access Control</title>
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background:#f8f9fa; }
        h1 { color: #222; }
        table { width: 100%; border-collapse: collapse; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.1); border-radius: 8px; overflow: hidden; }
        th, td { padding: 14px 16px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f1f3f4; font-weight: 600; color: #444; }
        tr:last-child td { border-bottom: none; }
        .status-granted { color: #2e7d32; font-weight: 600; }
        .status-denied  { color: #c62828; font-weight: 600; }
        .btn {
            padding: 6px 14px; border: none; border-radius: 4px; cursor: pointer;
            font-weight: 600; font-size: 0.9rem;
        }
        .btn-grant   { background: #2e7d32; color: #fff; }
        .btn-revoke  { background: #c62828; color: #fff; }
        .refresh { margin-top: 18px; color: #666; font-size: 0.85rem; }
        .empty { background: #fff; padding: 30px; border-radius: 8px; text-align: center; color: #555; }
    </style>
</head>
<body>
    <h1>🚪 Door Access Control</h1>
    <p>Toggle authorization below. Changes are picked up by the camera loop within <strong>{{ interval }}s</strong>.</p>

    {% if employees %}
    <table>
        <tr>
            <th>Employee</th>
            <th>Status</th>
            <th>Action</th>
        </tr>
        {% for name, authorized in employees %}
        <tr>
            <td>{{ name }}</td>
            <td class="{{ 'status-granted' if authorized else 'status-denied' }}">
                {{ '✓ Authorized' if authorized else '✗ Denied' }}
            </td>
            <td>
                <button class="btn {{ 'btn-revoke' if authorized else 'btn-grant' }}"
                        onclick="toggle('{{ name }}', {{ 'false' if authorized else 'true' }})">
                    {{ 'Revoke' if authorized else 'Grant' }}
                </button>
            </td>
        </tr>
        {% endfor %}
    </table>
    <p class="refresh">Auto-reload interval: {{ interval }}s</p>
    {% else %}
    <div class="empty">
        <p>No enrolled employees found.</p>
        <p>Run <code>python enroll.py</code> first.</p>
    </div>
    {% endif %}

    <script>
        function toggle(name, authorized) {
            fetch('/api/toggle', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name: name, authorized: authorized})
            })
            .then(r => r.json())
            .then(data => { if (data.success) location.reload(); else alert(data.error); });
        }
    </script>
</body>
</html>
"""

# ------------------------------------------------------------------ routes
@app.route("/")
@requires_auth
def index():
    employees = {}
    if os.path.exists(config.EMPLOYEES_JSON):
        with open(config.EMPLOYEES_JSON) as f:
            employees = json.load(f)

    auth = load_auth_csv()
    changed = False
    for name in employees:
        if name not in auth:
            auth[name] = config.DEFAULT_AUTHORIZED_IF_MISSING
            changed = True
    if changed:
        save_auth_csv(auth)

    employee_list = [
        (name, auth.get(name, config.DEFAULT_AUTHORIZED_IF_MISSING))
        for name in sorted(employees.keys())
    ]
    return render_template_string(
        HTML_TEMPLATE,
        employees=employee_list,
        interval=config.AUTH_RELOAD_INTERVAL_SECONDS,
    )


@app.route("/api/toggle", methods=["POST"])
@requires_auth
def toggle():
    data = request.get_json(force=True) or {}
    name = str(data.get("name", "")).strip()
    authorized = bool(data.get("authorized", False))

    if not name:
        return jsonify({"success": False, "error": "Missing name"}), 400

    auth = load_auth_csv()
    auth[name] = authorized
    save_auth_csv(auth)
    return jsonify({"success": True})


if __name__ == "__main__":
    if not os.path.exists(config.AUTH_CSV_FILE):
        save_auth_csv({})
    app.run(host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT, debug=False)
