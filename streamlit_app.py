import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Live Federated Learning Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.main-header {
    font-size: 2.2rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 0.25rem;
}
.sub-header {
    text-align: center;
    color: #666;
    margin-top: 0;
    margin-bottom: 1.25rem;
}
.status-running { color: #28a745; font-weight: bold; }
.status-waiting { color: #ffc107; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<h1 class="main-header">Live Federated Learning Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time privacy-preserving flood detection AI</p>', unsafe_allow_html=True)

st.sidebar.header("Dashboard Controls")
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 2)

if "monitor" not in st.session_state:
    st.session_state.monitor = auto_refresh

if st.sidebar.button("Start Monitor"):
    st.session_state.monitor = True
if st.sidebar.button("Stop Monitor"):
    st.session_state.monitor = False

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")

DEFAULT_METRICS_PATH = Path("metrics.json")
LATEST_METRICS_POINTER = Path("latest_metrics_path.txt")
status_placeholder = st.sidebar.empty()
metrics_placeholder = st.sidebar.empty()
countdown_placeholder = st.sidebar.empty()
main_container = st.container()


def resolve_metrics_path() -> Path:
    env_path = os.environ.get("METRICS_PATH")
    if env_path:
        return Path(env_path)
    if LATEST_METRICS_POINTER.exists():
        try:
            pointer_value = LATEST_METRICS_POINTER.read_text(encoding="utf-8").strip()
            if pointer_value:
                return Path(pointer_value)
        except Exception:
            pass
    return DEFAULT_METRICS_PATH


def load_metrics():
    metrics_path = resolve_metrics_path()
    if not metrics_path.exists():
        return {}
    try:
        with metrics_path.open(encoding="utf-8") as f:
            raw = f.read().strip()
            if not raw:
                return {}
            data = json.loads(raw)
        if "accuracies" not in data or not isinstance(data["accuracies"], list):
            return {}
        return data
    except Exception as e:
        status_placeholder.error(f"Error reading metrics.json: {e}")
        return {}


def format_pct(x):
    try:
        return f"{x:.1%}"
    except Exception:
        return str(x)


def display_dashboard(data):
    accuracies = data.get("accuracies", [])
    client_accuracies = data.get("client_accuracies", accuracies)
    rf_accuracies = data.get("rf_accuracies", [])
    status = data.get("status")
    last_updated = data.get("last_updated")
    rounds_expected = int(data.get("rounds_expected", 0) or 0)
    current_round = int(data.get("round_num", data.get("last_round", len(accuracies))) or 0)
    training_complete = bool(
        data.get("training_complete", False)
        or (rounds_expected > 0 and current_round >= rounds_expected and status == "completed")
    )

    if not accuracies:
        if status == "started":
            status_placeholder.markdown(
                f'<p class="status-running">Started - Round {current_round}</p>',
                unsafe_allow_html=True,
            )
            with main_container:
                st.info(f"Federated learning active. Round {current_round} is being prepared.")
            return False, accuracies

        status_placeholder.markdown(
            '<p class="status-waiting">Waiting for federated learning to start...</p>',
            unsafe_allow_html=True,
        )
        with main_container:
            st.info("Ready to start. Run the training launcher to begin.")
        with metrics_placeholder:
            st.metric("Latest Accuracy", "-")
            st.metric("Rounds Completed", "0")
        return False, accuracies

    current_acc = accuracies[-1]
    best_acc = max(accuracies)
    rounds_completed = current_round if current_round > 0 else len(accuracies)

    if training_complete:
        status_placeholder.markdown(
            '<p class="status-running">Training Complete</p>',
            unsafe_allow_html=True,
        )
    else:
        status_placeholder.markdown(
            f'<p class="status-running">Active - Round {current_round}</p>',
            unsafe_allow_html=True,
        )

    with metrics_placeholder:
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Current Round", current_round)
            st.metric("Latest Accuracy", format_pct(current_acc))
        with col_b:
            if len(accuracies) > 1:
                st.metric("Total Improvement", format_pct(current_acc - accuracies[0]))
            else:
                st.metric("Total Improvement", "-")

        if last_updated:
            try:
                dt = datetime.fromisoformat(last_updated)
                st.caption(f"Last update: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception:
                st.caption(f"Last update: {last_updated}")

    with main_container:
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.subheader("Federated Learning Progress")
            st.caption("Client Accuracy updates every round. RF Accuracy only changes on rounds where server-side RF evaluation runs.")
            df = pd.DataFrame(
                {
                    "Round": list(range(1, len(accuracies) + 1)),
                    "Client Accuracy (live every round)": client_accuracies,
                    "RF Accuracy (evaluated selectively)": rf_accuracies if rf_accuracies else [None] * len(accuracies),
                }
            ).set_index("Round")
            st.line_chart(df, height=420)

            st.subheader("Round-by-Round Details")
            for idx, acc in enumerate(accuracies, start=1):
                client_acc = client_accuracies[idx - 1] if idx - 1 < len(client_accuracies) else None
                rf_acc = rf_accuracies[idx - 1] if idx - 1 < len(rf_accuracies) else None
                if idx == 1:
                    st.write(
                        f"**Round {idx}:** client={format_pct(client_acc)}"
                        + (f", rf={format_pct(rf_acc)}" if rf_acc is not None else "")
                    )
                else:
                    prev_acc = client_accuracies[idx - 2] if idx - 2 < len(client_accuracies) else accuracies[idx - 2]
                    change = (client_acc - prev_acc) if client_acc is not None and prev_acc is not None else 0.0
                    direction = "up" if change > 0 else "flat" if change == 0 else "down"
                    st.write(
                        f"**Round {idx}:** client={format_pct(client_acc)} ({direction}, {change:+.1%})"
                        + (f", rf={format_pct(rf_acc)}" if rf_acc is not None else "")
                    )

        with col2:
            st.subheader("Key Metrics")
            st.metric("Current Accuracy", format_pct(current_acc), f"{current_acc:.4f}")
            st.metric("Best Accuracy", format_pct(best_acc))
            st.metric("Rounds Completed", rounds_completed)
            if training_complete:
                st.success("Training complete")
            else:
                st.info("Training in progress...")

        with col3:
            st.subheader("Run Info")
            st.metric("Clients", str(data.get("num_clients", 3)))
            st.metric("Rounds Expected", str(rounds_expected or "-"))
            st.metric("RF Evaluated", "Yes" if data.get("rf_evaluated") else "No")
            if data.get("latest_client_accuracy") is not None:
                st.metric("Latest Client Accuracy", format_pct(data.get("latest_client_accuracy")))
            if rf_accuracies:
                st.metric("Latest RF Accuracy", format_pct(rf_accuracies[-1]))

            st.subheader("Export Data")
            st.download_button(
                "Download Metrics",
                data=json.dumps(data, indent=2),
                file_name="federated_learning_results.json",
                mime="application/json",
            )

            if st.checkbox("Show Raw Data"):
                st.json(data)

    return training_complete, accuracies


data = load_metrics()
is_complete, accuracies = display_dashboard(data)

if st.session_state.monitor and auto_refresh:
    if is_complete:
        countdown_placeholder.success("Training complete. Auto-refresh paused.")
        st.session_state.monitor = False
    else:
        for i in range(refresh_rate, 0, -1):
            countdown_placeholder.metric("Next Refresh", f"{i}s")
            time.sleep(1)
        countdown_placeholder.empty()
        st.rerun()
else:
    active_metrics_path = resolve_metrics_path()
    if not active_metrics_path.exists():
        st.info("metrics.json not found. Run the training launcher first.")
    elif accuracies:
        st.caption(f"Snapshot: {len(accuracies)} datapoints from {active_metrics_path.name}. Toggle Start Monitor for live updates.")
