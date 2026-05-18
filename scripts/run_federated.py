#!/usr/bin/env python3
"""One-command launcher for the Flower server and local clients.

Starts the server, waits until the address is reachable, launches the requested
number of clients, and streams prefixed logs in a single terminal.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
LATEST_METRICS_POINTER = REPO_ROOT / "latest_metrics_path.txt"
BEST_METRICS_PATH = REPO_ROOT / "best_metrics.json"
BEST_ARTIFACT_DIR = REPO_ROOT / "best_artifacts"


def _parse_progressive_schedule(schedule: str) -> dict[int, int]:
    parsed: dict[int, int] = {}
    if not schedule:
        return parsed
    for entry in schedule.split(","):
        item = entry.strip()
        if not item:
            continue
        round_str, blocks_str = item.split(":", 1)
        parsed[int(round_str)] = int(blocks_str)
    return parsed


def _blocks_for_round(default_blocks: int, schedule: dict[int, int], server_round: int) -> int:
    active_blocks = int(default_blocks)
    for round_idx in sorted(schedule):
        if server_round >= round_idx:
            active_blocks = int(schedule[round_idx])
        else:
            break
    return active_blocks


def _reader_thread(stream, tag: str, output_queue: "queue.Queue[Tuple[str, str]]") -> None:
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            output_queue.put((tag, line.rstrip()))
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _wait_for_address(host: str, port: int, timeout_sec: float) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                sock.connect((host, port))
                return True
            except OSError:
                time.sleep(0.5)
    return False


def _spawn_process(tag: str, args: List[str]) -> Tuple[subprocess.Popen, threading.Thread]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        args,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    output_queue = _spawn_process.output_queue
    thread = threading.Thread(
        target=_reader_thread,
        args=(process.stdout, tag, output_queue),
        daemon=True,
    )
    thread.start()
    return process, thread


_spawn_process.output_queue = queue.Queue()


def _drain_logs(processes: List[Tuple[str, subprocess.Popen]], timeout: float = 0.2) -> None:
    output_queue = _spawn_process.output_queue
    drained = False
    while True:
        try:
            tag, line = output_queue.get(timeout=timeout if not drained else 0.0)
        except queue.Empty:
            break
        print(f"[{tag}] {line}")
        drained = True


def _terminate_processes(processes: List[Tuple[str, subprocess.Popen]]) -> None:
    for _tag, process in reversed(processes):
        if process.poll() is None:
            process.terminate()

    deadline = time.time() + 10
    while time.time() < deadline:
        alive = [process for _tag, process in processes if process.poll() is None]
        if not alive:
            return
        time.sleep(0.2)

    for _tag, process in reversed(processes):
        if process.poll() is None:
            process.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Flower server + clients from one terminal.")
    parser.add_argument("--num_clients", type=int, default=3, help="Number of local Flower clients to launch.")
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of federated rounds.")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for server and clients.")
    parser.add_argument("--max_batches_per_round", type=int, default=0, help="Cap local client training to this many batches each round. 0 uses all batches.")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", choices=["simplecnn", "efficientnet_b0"])
    parser.add_argument("--client_selection", type=str, default="all", choices=["all", "sampled"], help="Use all launched clients every round, or sampled participation.")
    parser.add_argument("--trainable_blocks", type=int, default=1, help="Number of EfficientNet feature blocks to federate.")
    parser.add_argument("--progressive_unfreeze_schedule", type=str, default="", help="Comma-separated round:block schedule, e.g. '1:1,3:2,5:3'.")
    parser.add_argument("--rf_eval_interval", type=int, default=0, help="0 means final-round-only RF eval, otherwise every N rounds.")
    parser.add_argument("--fraction_fit", type=float, default=1.0, help="Fraction of clients sampled each round.")
    parser.add_argument("--min_fit_clients", type=int, default=None, help="Minimum clients selected per round. Defaults to all launched clients.")
    parser.add_argument("--min_available_clients", type=int, default=None, help="Minimum connected clients required before training starts. Defaults to all launched clients.")
    parser.add_argument("--address", type=str, default="127.0.0.1:8080", help="Flower gRPC address.")
    parser.add_argument("--startup_timeout", type=float, default=60.0, help="Seconds to wait for server readiness.")
    args = parser.parse_args()

    python_exe = sys.executable
    host, port_str = args.address.rsplit(":", 1)
    port = int(port_str)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    metrics_out = RUNS_DIR / f"metrics_{run_id}.json"
    run_artifact_dir = RUNS_DIR / f"artifacts_{run_id}"
    with LATEST_METRICS_POINTER.open("w", encoding="utf-8") as handle:
        handle.write(str(metrics_out))
    rf_eval_interval = args.rf_eval_interval if args.rf_eval_interval > 0 else args.num_rounds
    progressive_schedule = _parse_progressive_schedule(args.progressive_unfreeze_schedule)
    initial_trainable_blocks = _blocks_for_round(args.trainable_blocks, progressive_schedule, 1)
    max_trainable_blocks = max(progressive_schedule.values(), default=initial_trainable_blocks)
    max_trainable_blocks = max(max_trainable_blocks, initial_trainable_blocks)
    if args.client_selection == "all":
        fraction_fit = 1.0
        min_fit_clients = args.num_clients
        min_available_clients = args.num_clients
    else:
        fraction_fit = args.fraction_fit
        min_fit_clients = args.min_fit_clients if args.min_fit_clients is not None else max(2, int(round(args.num_clients * fraction_fit)))
        min_available_clients = (
            args.min_available_clients if args.min_available_clients is not None else args.num_clients
        )

    processes: List[Tuple[str, subprocess.Popen]] = []
    threads: List[threading.Thread] = []

    server_cmd = [
        python_exe,
        "-u",
        "server.py",
        "--backbone",
        args.backbone,
        "--client_selection",
        args.client_selection,
        "--num_clients",
        str(args.num_clients),
        "--num_rounds",
        str(args.num_rounds),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--max_batches_per_round",
        str(args.max_batches_per_round),
        "--progressive_unfreeze_schedule",
        args.progressive_unfreeze_schedule,
        "--address",
        args.address,
        "--metrics_out",
        str(metrics_out),
        "--best_metrics_out",
        str(BEST_METRICS_PATH),
        "--run_artifact_dir",
        str(run_artifact_dir),
        "--best_artifact_dir",
        str(BEST_ARTIFACT_DIR),
        "--rf_eval_interval",
        str(rf_eval_interval),
        "--fraction_fit",
        str(fraction_fit),
        "--min_fit_clients",
        str(min_fit_clients),
        "--min_available_clients",
        str(min_available_clients),
        "--trainable_blocks",
        str(initial_trainable_blocks),
        "--max_trainable_blocks",
        str(max_trainable_blocks),
        "--no-initial_eval",
    ]

    print("[launcher] Starting Flower server...")
    print(
        f"[launcher] Config: clients={args.num_clients} | client_selection={args.client_selection} | fraction_fit={fraction_fit} | "
        f"min_fit_clients={min_fit_clients} | min_available_clients={min_available_clients} | "
        f"max_batches_per_round={args.max_batches_per_round} | trainable_blocks={initial_trainable_blocks}/{max_trainable_blocks} | "
        f"progressive_schedule={args.progressive_unfreeze_schedule or 'off'} | metrics_out={metrics_out} | "
        f"best_metrics_out={BEST_METRICS_PATH} | run_artifact_dir={run_artifact_dir} | "
        f"best_artifact_dir={BEST_ARTIFACT_DIR}"
    )
    server_process, server_thread = _spawn_process("server", server_cmd)
    processes.append(("server", server_process))
    threads.append(server_thread)

    try:
        while True:
            _drain_logs(processes)
            if server_process.poll() is not None:
                print(f"[launcher] Server exited early with code {server_process.returncode}")
                return server_process.returncode or 1
            if _wait_for_address(host, port, timeout_sec=0.5):
                break
            if args.startup_timeout <= 0:
                print("[launcher] Server did not become reachable in time.")
                return 1
            args.startup_timeout -= 0.5

        print("[launcher] Server is ready. Starting clients...")
        for cid in range(1, args.num_clients + 1):
            client_cmd = [
                python_exe,
                "-u",
                "client.py",
                "--cid",
                str(cid),
                "--backbone",
                args.backbone,
                "--train_backbone",
                "--trainable_blocks",
                str(initial_trainable_blocks),
                "--max_trainable_blocks",
                str(max_trainable_blocks),
                "--batch_size",
                str(args.batch_size),
                "--address",
                args.address,
            ]
            process, thread = _spawn_process(f"client-{cid}", client_cmd)
            processes.append((f"client-{cid}", process))
            threads.append(thread)

        while True:
            _drain_logs(processes)
            server_return = server_process.poll()
            if server_return is not None:
                summary_path = RUNS_DIR / f"run_{run_id}.json"
                summary = {
                    "run_id": run_id,
                    "metrics_path": str(metrics_out),
                    "artifact_dir": str(run_artifact_dir),
                    "server_exit_code": server_return,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                with summary_path.open("w", encoding="utf-8") as handle:
                    json.dump(summary, handle, indent=2)
                print(f"[launcher] Server exited with code {server_return}. Shutting down clients...")
                return server_return

            failed_clients = [
                (tag, process.returncode)
                for tag, process in processes[1:]
                if process.poll() not in (None, 0)
            ]
            if failed_clients:
                for tag, returncode in failed_clients:
                    print(f"[launcher] {tag} exited unexpectedly with code {returncode}")
                return 1

            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[launcher] Interrupt received. Stopping server and clients...")
        return 130
    finally:
        _terminate_processes(processes)
        _drain_logs(processes, timeout=0.05)
        for thread in threads:
            thread.join(timeout=0.2)


if __name__ == "__main__":
    if os.name == "nt":
        signal.signal(signal.SIGINT, signal.default_int_handler)
    raise SystemExit(main())
