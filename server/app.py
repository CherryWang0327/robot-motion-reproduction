from __future__ import annotations

import argparse
import json
import mimetypes
import re
import shutil
import subprocess
import sys
import threading
import uuid
from html import escape
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from . import db

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
UPLOADS = ROOT / "uploads"
STATIC = ROOT / "web"
JOBS: dict[str, subprocess.Popen] = {}
SAFE_NAME = re.compile(r"[^a-zA-Z0-9_-]+")


def safe_run(name: str) -> Path:
    path = (RUNS / name).resolve()
    if path.parent != RUNS.resolve():
        raise ValueError("invalid run")
    return path


def stage_data(run: Path) -> dict:
    stages = []
    for stage in sorted(run.rglob("stage.json")):
        try:
            data = json.loads(stage.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        data["path"] = str(stage.relative_to(run))
        stages.append(data)
    report = run / "04_validation" / "report.json"
    if report.is_file():
        try:
            validation = json.loads(report.read_text(encoding="utf-8"))
            stages.append({"stage": "validation", "status": validation.get("status", "FAIL"), "path": "04_validation/report.json"})
        except (OSError, json.JSONDecodeError):
            pass
    running = run.name in JOBS and JOBS[run.name].poll() is None
    statuses = {stage.get("status") for stage in stages}
    if running:
        status = "RUNNING"
    elif "FAIL" in statuses:
        status = "FAIL"
    elif any(stage.get("stage") == "convert_to_wbt" and stage.get("status") == "PASS" for stage in stages):
        status = "COMPLETE"
    elif stages:
        status = "PLANNED" if statuses == {"DRY_RUN"} else "WAITING"
    else:
        status = "WAITING"
    return {"name": run.name, "stages": stages, "status": status}


def dashboard_items() -> list[dict]:
    """Return dashboard rows in the same order for API and no-JS HTML."""
    RUNS.mkdir(exist_ok=True)
    jobs = {item["name"]: item for item in db.all_jobs()}
    data = [stage_data(run) for run in RUNS.iterdir() if run.is_dir()]
    known = {item["name"] for item in data}
    for name, job in jobs.items():
        if name in known:
            item = next(item for item in data if item["name"] == name)
            item["status"] = job["status"]
            item["route"] = job["route"]
            item["error"] = job["error"]
        else:
            data.append({"name": name, "stages": [], "status": job["status"], "route": job["route"], "error": job["error"]})
    data.sort(key=lambda item: jobs.get(item["name"], {}).get("updated_at", ""), reverse=True)
    data.sort(key=lambda item: item.get("status") == "RUNNING", reverse=True)
    return data


def dashboard_html() -> str:
    items = dashboard_items()
    if not items:
        return '<p class="empty">还没有任务。上传一个视频开始。</p>'
    return "".join(
        f'<article class="run"><div class="run-head"><h3>{escape(item["name"])}</h3>'
        f'<span class="state state-{escape(item.get("status", "WAITING").lower())}">{escape(item.get("status", "WAITING"))}</span>'
        f'</div><p class="sub">服务端实时状态。加载完成后可展开各阶段与结果。</p></article>'
        for item in items
    )


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def _json(self, value: object, status: int = 200) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def end_headers(self) -> None:
        # The dashboard polls live jobs; stale HTML or JavaScript makes a
        # newly created running task appear to be missing.
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def do_GET(self) -> None:
        request = urlparse(self.path)
        if request.path == "/api/runs":
            return self._json(dashboard_items())
        if request.path.startswith("/api/runs/"):
            try:
                return self._json(stage_data(safe_run(request.path.rsplit("/", 1)[-1])))
            except (ValueError, FileNotFoundError):
                return self._json({"error": "run not found"}, 404)
        if request.path.startswith("/api/results/"):
            try:
                run = safe_run(request.path.rsplit("/", 1)[-1])
                stage = parse_qs(request.query).get("stage", [""])[0]
                base = (run / stage).resolve() if stage else run
                if run != base and run not in base.parents:
                    raise ValueError("invalid stage")
                allowed = {".mp4", ".webm", ".json", ".csv", ".log", ".npz", ".pt", ".pkl"}
                files = [str(path.relative_to(ROOT)) for path in sorted(base.rglob("*")) if path.is_file() and path.suffix.lower() in allowed]
                return self._json({"files": files})
            except (ValueError, FileNotFoundError):
                return self._json({"error": "run not found"}, 404)
        if request.path == "/api/file":
            value = parse_qs(request.query).get("path", [""])[0]
            path = (ROOT / value).resolve()
            if not path.is_file() or ROOT not in path.parents:
                return self._json({"error": "file not found"}, 404)
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as stream:
                shutil.copyfileobj(stream, self.wfile)
            return
        if request.path == "/":
            page = (STATIC / "index.html").read_text(encoding="utf-8")
            page = page.replace('<div id="runs" class="runs"></div>', f'<div id="runs" class="runs">{dashboard_html()}</div>')
            encoded = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        return super().do_GET()

    def do_POST(self) -> None:
        request = urlparse(self.path)
        if request.path == "/api/upload":
            name = self.headers.get("X-Filename", "video.mp4")
            suffix = Path(name).suffix.lower()
            if suffix not in {".mp4", ".mov", ".avi", ".mkv"}:
                return self._json({"error": "unsupported video type"}, 400)
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2 * 1024**3:
                return self._json({"error": "video must be between 1 byte and 2 GB"}, 400)
            UPLOADS.mkdir(exist_ok=True)
            saved = UPLOADS / f"{uuid.uuid4().hex}{suffix}"
            with saved.open("wb") as stream:
                stream.write(self.rfile.read(length))
            return self._json({"video": str(saved.relative_to(ROOT))})
        if request.path == "/api/build":
            size = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(size))
            route = body.get("route")
            video = (ROOT / body.get("video", "")).resolve()
            name = SAFE_NAME.sub("-", body.get("name", "motion")).strip("-")[:48]
            if route not in {"gmr", "protomotions"} or not video.is_file() or not name:
                return self._json({"error": "invalid route, video, or run name"}, 400)
            if name in JOBS and JOBS[name].poll() is None:
                return self._json({"error": "this run is already active"}, 409)
            run = safe_run(name)
            if run.exists():
                return self._json({"error": "run name already exists"}, 409)
            (run / "00_job").mkdir(parents=True)
            db.create(name, str(video.relative_to(ROOT)), route)
            command = [sys.executable, "-m", "server.worker", "--name", name]
            log = (run / "00_job" / "worker-launcher.log").open("w", encoding="utf-8")
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            JOBS[name] = process
            (run / "00_job" / "job.json").write_text(json.dumps({"command": command, "pid": process.pid, "safety": "no training; no robot access"}, indent=2) + "\n")
            return self._json({"name": name, "status": "started"}, 202)
        if request.path.startswith("/api/approve/"):
            name = request.path.rsplit("/", 1)[-1]
            try:
                run = safe_run(name)
                valid = any(json.loads(path.read_text()).get("stage") == "convert_to_wbt" and json.loads(path.read_text()).get("status") == "PASS" for path in run.rglob("stage.json"))
                if not valid:
                    return self._json({"error": "WBT 验收尚未通过，不能进入训练队列"}, 409)
                db.update(name, "TRAINING_QUEUED")
                return self._json({"status": "TRAINING_QUEUED"})
            except (ValueError, FileNotFoundError):
                return self._json({"error": "run not found"}, 404)
        if request.path.startswith("/api/cancel/"):
            name = request.path.rsplit("/", 1)[-1]
            try:
                safe_run(name)
                process = JOBS.get(name)
                if process and process.poll() is None:
                    process.terminate()
                db.update(name, "CANCELLED", "Stopped by user")
                return self._json({"status": "CANCELLED"})
            except ValueError:
                return self._json({"error": "run not found"}, 404)
        return self._json({"error": "unknown endpoint"}, 404)


def main() -> None:
    parser = argparse.ArgumentParser(description="G1 Motion Pipeline local web console")
    parser.add_argument("--host", default="127.0.0.1", help="Keep localhost unless LAN exposure is explicitly intended")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
