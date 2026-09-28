"""
Main Entry Point & Server Launcher for VeriShield AI
Runs dataset initialization, model verification, smart port allocation,
and launches the FastAPI Web Server + Interactive Dashboard.
"""
import os
import sys
import socket
import argparse
import threading
import webbrowser
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import uvicorn
from dataset.generate_dataset import generate_csv, DATASET_PATH
from src.train import run_training_pipeline, MODEL_ARTIFACT_PATH


def is_port_in_use(host: str, port: int) -> bool:
    """Checks whether a given host/port socket is currently bound/in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def find_available_port(host: str, start_port: int = 8000, max_attempts: int = 20) -> int:
    """Finds the first available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(host, port):
            return port
    return start_port


def open_browser(url: str, delay_seconds: float = 1.2):
    """Opens the user's default browser to the web dashboard after a slight delay."""
    def _open():
        try:
            webbrowser.open(url)
        except Exception:
            pass
    threading.Timer(delay_seconds, _open).start()


def main():
    parser = argparse.ArgumentParser(description="VeriShield AI Server Launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Host IP to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open web browser")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload for development")
    args = parser.parse_args()

    print("=" * 70)
    print("🛡️  VeriShield AI - Hybrid Fake News Detection & Explainable AI")
    print("🏛️  Vels Institute of Science, Technology & Advanced Studies (VISTAS)")
    print("👨‍💻 Authors: Mohamed Irfaan M, Bhuvanesh M, Harish Hariharan MU")
    print("=" * 70)

    try:
        # 1. Ensure benchmark dataset exists
        if not os.path.exists(DATASET_PATH):
            print("\n[Init] Generating benchmark dataset...")
            generate_csv()

        # 2. Ensure AI model and XAI artifacts exist
        if not os.path.exists(MODEL_ARTIFACT_PATH):
            print("\n[Init] Training Hybrid AI Model & generating Explainable AI artifacts...")
            run_training_pipeline()
        else:
            print("\n[Init] Model artifacts verified in models/ directory.")

        # 3. Resolve Port & Detect Port Conflicts
        target_port = args.port
        if is_port_in_use(args.host, target_port):
            available_port = find_available_port(args.host, start_port=target_port)
            print(f"\n⚠️  Port {target_port} is already in use by another running process.")
            print(f"🔄 Automatically re-routing VeriShield to available port: {available_port}")
            target_port = available_port

        dash_url = f"http://localhost:{target_port}"
        docs_url = f"http://localhost:{target_port}/docs"

        print("\n" + "=" * 70)
        print(f"🌐 VeriShield Web Dashboard: {dash_url}")
        print(f"📖 Interactive API Docs:     {docs_url}")
        print("=" * 70 + "\n")

        # 4. Automatically open web dashboard in browser
        if not args.no_browser:
            open_browser(dash_url)

        # 5. Launch FastAPI server via Uvicorn
        uvicorn.run("backend.app:app", host=args.host, port=target_port, reload=args.reload)

    except KeyboardInterrupt:
        print("\n\n🛑 VeriShield AI Server stopped cleanly by user.")
    except Exception as e:
        print("\n" + "!" * 70)
        print("❌ CRITICAL ERROR during startup:")
        traceback.print_exc()
        print("!" * 70)
        # Keep window open if run via double-click
        try:
            input("\nPress [Enter] to exit...")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
