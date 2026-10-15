"""Local browser dashboard for exploring TCLAD-FL experiment outputs."""
from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
WEB = ROOT / "web"
RUNS: dict[str, dict[str, object]] = {}
LOCK = threading.Lock()


def safe_file(relative: str) -> Path:
    target = (ROOT / unquote(relative)).resolve()
    if target != ROOT and ROOT not in target.parents:
        raise ValueError("Path is outside the project")
    return target


def experiment_data() -> dict[str, object]:
    experiments: list[dict[str, object]] = []
    raw = RESULTS / "raw" / "federated"
    if raw.exists():
        for metrics_path in sorted(raw.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                history = payload.get("history", [])
                last = history[-1] if history else {}
                experiments.append({
                    "id": payload.get("experiment_id", metrics_path.parent.name),
                    "path": metrics_path.parent.relative_to(ROOT).as_posix(),
                    "seed": payload.get("seed"),
                    "device": payload.get("device"),
                    "attack": payload.get("attack_name", "none"),
                    "aggregation": payload.get("aggregation", "fedavg"),
                    "partition": payload.get("partition", {}).get("type", "unknown"),
                    "rounds": len(history),
                    "accuracy": last.get("global_accuracy"),
                    "loss": last.get("global_loss"),
                    "asr": last.get("asr"),
                    "runtime": payload.get("runtime_seconds"),
                    "history": [{
                        "round": row.get("round"),
                        "accuracy": row.get("global_accuracy"),
                        "loss": row.get("global_loss"),
                        "asr": row.get("asr"),
                    } for row in history],
                })
            except (OSError, ValueError, TypeError):
                continue
    figures = sorted(p.relative_to(ROOT).as_posix() for p in (RESULTS / "figures").glob("*.png")) if (RESULTS / "figures").exists() else []
    tables = sorted(p.relative_to(ROOT).as_posix() for p in (RESULTS / "tables").glob("*.csv")) if (RESULTS / "tables").exists() else []
    return {"experiments": experiments, "figures": figures, "tables": tables, "runs": RUNS}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def log_message(self, fmt: str, *args: object) -> None:
        print("[web] " + fmt % args)

    def send_json(self, data: object, status: int = 200) -> None:
        encoded = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/data":
            self.send_json(experiment_data())
            return
        if parsed.path.startswith("/files/"):
            try:
                file_path = safe_file(parsed.path.removeprefix("/files/"))
                if not file_path.is_file():
                    self.send_error(404)
                    return
                content = file_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(file_path.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except ValueError:
                self.send_error(403)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/run":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(size))
            kind = request.get("kind")
            config_name = request.get("config")
            allowed_configs = {"mnist.yaml", "mnist_lsa_late.yaml", "mnist_conventional_late.yaml"}
            if kind not in {"baseline", "lsa", "conventional"} or config_name not in allowed_configs:
                self.send_json({"error": "Choose a supported run type and configuration."}, 400)
                return
            script = "train_federated.py" if kind == "baseline" else "run_lsa.py"
            experiment_id = f"web-{kind}-{__import__('time').strftime('%Y%m%d-%H%M%S')}"
            python = Path(sys.executable)
            command = [str(python), str(ROOT / "scripts" / script), "--config", str(ROOT / "configs" / config_name), "--experiment-id", experiment_id]
            if kind == "conventional":
                command[command.index(str(ROOT / "configs" / config_name))] = str(ROOT / "configs" / "mnist_conventional_late.yaml")
            with LOCK:
                RUNS[experiment_id] = {"status": "running", "kind": kind, "output": "", "error": None}
            threading.Thread(target=self.run_process, args=(experiment_id, command), daemon=True).start()
            self.send_json({"experiment_id": experiment_id, "status": "running"}, 202)
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 400)

    @staticmethod
    def run_process(experiment_id: str, command: list[str]) -> None:
        try:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONUNBUFFERED": "1"})
            output = ""
            assert proc.stdout is not None
            for line in proc.stdout:
                output = (output + line)[-12000:]
                with LOCK:
                    RUNS[experiment_id]["output"] = output
            code = proc.wait()
            with LOCK:
                RUNS[experiment_id]["status"] = "completed" if code == 0 else "failed"
                RUNS[experiment_id]["return_code"] = code
        except OSError as exc:
            with LOCK:
                RUNS[experiment_id].update(status="failed", error=str(exc))


def main() -> None:
    port = int(os.environ.get("TCLAD_DEMO_PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"TCLAD-FL demo is running at {url}")
    print("Only your computer can access this local demo. Press Ctrl+C here to stop it.")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping TCLAD-FL demo.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

