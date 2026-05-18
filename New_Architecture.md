# Current Architecture: Federated Disaster Image Classification (Low-Latency EfficientNet-B0 Path)

## Overview

This document describes the **current production-style architecture** of the project after the latency, orchestration, metrics, and experiment-management improvements.

The system remains a **Flower-based Federated Learning (FL)** workflow, but the engineering has evolved significantly from the earlier design:

1. **Federated EfficientNet-B0 Backbone**
   - The shared model is an EfficientNet-B0 feature extractor.
   - Only a selected trainable slice of the backbone is communicated and aggregated.
   - The trainable slice can grow over time through **progressive unfreezing**.

2. **Client-Private Neural Head**
   - Each client keeps its own `LocalHead(feature_dim -> num_classes)`.
   - This head is used for supervised local learning and local evaluation.
   - It is never shared with the server.

3. **Server-Side Global RF Evaluation**
   - The server uses the aggregated backbone as a feature extractor.
   - It trains a **PCA + RandomForest** pipeline on backbone features from all client training sets.
   - It evaluates that downstream RF on the held-out `global_test` set.

4. **Run-Scoped Metrics and Artifacts**
   - Every run writes its own metrics file and artifact directory.
   - The best run is preserved in dedicated `best_metrics.json` and `best_artifacts/`.
   - Lower-performing runs no longer overwrite the strongest checkpoint and downstream artifacts.

5. **One-Command Orchestration**
   - A single launcher starts the server, waits for readiness, starts all clients, and streams logs in one terminal.

This architecture aims to balance:

- **privacy**
- **federated correctness**
- **low latency**
- **repeatable experimentation**
- **artifact safety**
- **operational simplicity**

## Design Goals

The current architecture is engineered around the following goals:

- Keep the project truly federated: client data never leaves the node.
- Retain EfficientNet-B0 quality rather than downgrading to a lighter but weaker backbone.
- Reduce round latency without collapsing the FL objective.
- Make the system understandable and reproducible for demos, experiments, and mentoring discussions.
- Separate **live training progress** from **heavy downstream RF evaluation**.
- Preserve the best-performing artifacts across many experimental runs.

## High-Level Architecture

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                    CURRENT FEDERATED SYSTEM                             │
├──────────────────────────────────────────────────────────────────────────┤
│  SERVER (server.py)                                                     │
│  ├─ Flower FedAvg orchestration                                         │
│  ├─ EfficientNet-B0 shared backbone state                               │
│  ├─ Partial parameter communication (trainable slice only)              │
│  ├─ Progressive unfreezing schedule                                     │
│  ├─ Weighted aggregation of client-side live accuracy                   │
│  ├─ Throttled PCA + RandomForest evaluation                             │
│  ├─ Run-specific metrics + artifact persistence                         │
│  └─ Best-run checkpoint preservation                                    │
│                        │                                                 │
│                        │ gRPC via Flower                                 │
│                        │ selected backbone tensors only                  │
│                        ▼                                                 │
├──────────────────────────────────────────────────────────────────────────┤
│  CLIENTS (client.py × N)                                                │
│  ├─ Local private dataset                                               │
│  ├─ EfficientNet-B0 backbone                                            │
│  ├─ Client-private LocalHead                                            │
│  ├─ Local supervised training                                           │
│  ├─ Optional capped batches per round                                   │
│  ├─ Guaranteed full dataset coverage across rounds                      │
│  ├─ Local eval each round                                               │
│  └─ Return only shared backbone tensors                                 │
├──────────────────────────────────────────────────────────────────────────┤
│  DASHBOARD (streamlit_app.py)                                           │
│  ├─ Reads latest run metrics automatically                              │
│  ├─ Shows client accuracy every round                                   │
│  ├─ Shows RF accuracy on measured rounds                                │
│  ├─ Displays training status / rounds / metadata                        │
│  └─ Stops auto-refresh when training completes                          │
├──────────────────────────────────────────────────────────────────────────┤
│  LAUNCHER (scripts/run_federated.py + .ps1)                             │
│  ├─ Single command to run server + clients                              │
│  ├─ Creates run-specific metrics path                                   │
│  ├─ Creates run-specific artifact directory                             │
│  ├─ Updates latest metrics pointer                                      │
│  └─ Preserves global best artifacts                                     │
└──────────────────────────────────────────────────────────────────────────┘
```

## Exact Runtime Components

### 1. Server (`server.py`)

The server is the coordination and evaluation brain of the system.

Current responsibilities:

- Start Flower server with `FedAvg`
- Build the shared backbone
- Freeze/unfreeze the correct EfficientNet blocks
- Define the set of **communicated state tensors**
- Provide initial shared parameters to clients
- Send per-round config to clients:
  - epochs
  - batch size
  - max batches per round
  - active trainable blocks
- Aggregate client updates
- Aggregate client-reported live accuracy
- Run server-side RF evaluation on selected rounds
- Persist run metrics
- Preserve best metrics and best artifacts

### 2. Clients (`client.py`)

Each client simulates one private institutional node with local data.

Current responsibilities:

- Load `data/client_N/train` and `data/client_N/test`
- Build EfficientNet-B0 backbone
- Build a private local classification head
- Receive shared backbone tensors from the server
- Train local head and selected backbone slice
- Evaluate locally on client test data
- Return only shared backbone tensors to the server
- Log local dataset coverage and payload size

### 3. Data Pipeline (`dataset_loader.py`)

The loader is designed for practical robustness instead of just demo cleanliness.

Current behavior:

- Uses `ImageFolder` layout
- Handles corrupted/truncated images through `RobustImageFolder`
- Supports two transform presets:
  - `simplecnn`: `64x64`
  - `efficientnet_b0`: `224x224` + ImageNet normalization
- Uses CPU-safe dataloader settings on Windows
- Enables more optimized loading when environment supports it

### 4. Model Definitions (`models.py`)

Current neural components:

- `EfficientNetB0Extractor`
  - feature extractor only
  - classifier removed
  - output feature dimension = `1280`
- `LocalHead`
  - private linear classifier on clients
- `SimpleCNN`
  - still available for compatibility
  - no longer the preferred backbone for current experiments

### 5. Shared Utility Layer (`utils.py`)

Critical current functions:

- `get_device()`
  - picks `cuda`, then `mps`, then `cpu`
- `configure_backbone_training(...)`
  - freezes all params, then unfreezes only selected EfficientNet blocks
- `get_trainable_state_keys(...)`
  - finds exactly which state tensors are shared
- `get_parameters_from_model(...)`
  - serializes selected tensors
- `set_parameters_to_model(...)`
  - restores selected tensors
- `payload_size_kb(...)`
  - estimates communication cost

### 6. Dashboard (`streamlit_app.py`)

The dashboard now reflects the modern system more honestly.

Current behavior:

- Resolves metrics automatically using:
  1. `METRICS_PATH`
  2. `latest_metrics_path.txt`
  3. fallback `metrics.json`
- Shows:
  - current round
  - rounds expected
  - client accuracy
  - RF accuracy
  - training status
  - latest update time
- Uses separate lines for:
  - **Client Accuracy (live every round)**
  - **RF Accuracy (evaluated selectively)**
- Pauses auto-refresh at completion

### 7. Launcher (`scripts/run_federated.py`, `scripts/run_federated.ps1`)

The launcher is now the main user-facing operational interface.

Current behavior:

- Starts server first
- Waits until server port becomes reachable
- Starts all requested clients
- Streams logs from server and all clients into one terminal
- Creates:
  - `runs/metrics_YYYYMMDD_HHMMSS.json`
  - `runs/artifacts_YYYYMMDD_HHMMSS/`
  - `runs/run_YYYYMMDD_HHMMSS.json`
- Updates:
  - `latest_metrics_path.txt`
- Preserves:
  - `best_metrics.json`
  - `best_artifacts/`

## Current File Hierarchy (Relevant Operational Files)

```text
d:/Tech stuffs/flwr-flood-damage/
├── client.py
├── server.py
├── dataset_loader.py
├── models.py
├── utils.py
├── streamlit_app.py
├── Architecture.md
├── New_Architecture.md
├── improvements.md
├── metrics.json
├── metrics_idle.json
├── latest_metrics_path.txt
├── best_metrics.json
├── best_artifacts/
│   ├── global_backbone_best.pt
│   ├── global_rf_best.pkl
│   ├── global_pca_best.pkl
│   └── best_artifacts.json
├── runs/
│   ├── metrics_YYYYMMDD_HHMMSS.json
│   ├── run_YYYYMMDD_HHMMSS.json
│   └── artifacts_YYYYMMDD_HHMMSS/
│       ├── global_backbone_latest.pt
│       ├── global_backbone_best_in_run.pt
│       ├── global_backbone_final.pt
│       ├── global_rf.pkl
│       ├── global_pca.pkl
│       └── run_summary.json
└── scripts/
    ├── run_federated.py
    ├── run_federated.ps1
    ├── start_server.ps1
    └── start_clients.ps1
```

## Current Training Logic

### Federated Training Unit

The actual federated object is:

- the **selected trainable subset** of the EfficientNet-B0 backbone

The following are **not federated**:

- raw images
- local client classification heads
- server-side Random Forest
- server-side PCA

This means the project is still a genuine FL system:

- client data stays local
- shared model weights are aggregated centrally
- representation learning happens collaboratively

### Partial EfficientNet Fine-Tuning

The current system uses **partial backbone federation**.

Instead of always updating the full EfficientNet-B0, the system:

- freezes most of the backbone
- unfreezes only the last `N` EfficientNet feature blocks
- communicates only the state tensors belonging to those trainable blocks

Why this matters:

- reduces communication payload
- reduces client compute cost
- keeps pretrained early visual features stable
- still allows useful global adaptation

### Progressive Unfreezing

The system supports schedules such as:

```text
1:1,4:2,8:3
```

Meaning:

- rounds `1-3`: train last `1` EfficientNet block
- rounds `4-7`: train last `2` blocks
- rounds `8+`: train last `3` blocks

Why this matters:

- early rounds are cheaper and more stable
- later rounds allow deeper adaptation
- it improves accuracy without paying full end-to-end cost from round 1

### Local Dataset Coverage Across Rounds

The client no longer needs to process the full local dataset every round when speed matters.

If `max_batches_per_round > 0`:

- a shuffled sample order is created
- round 1 uses the first chunk
- round 2 uses the next chunk
- and so on
- only after the full dataset is covered is the order reshuffled

This gives:

- controlled latency
- guaranteed full coverage over successive rounds
- no accidental permanent starvation of some samples

## Data Flow for One Federated Round

```text
Round N

Server:
1. Select active trainable EfficientNet blocks
2. Send only shared state tensors for those blocks
3. Wait for client updates

Clients:
4. Load incoming shared backbone tensors
5. Keep local head private
6. Build round-specific local subset if max_batches_per_round > 0
7. Train backbone slice + local head
8. Evaluate locally on private client test set
9. Return updated shared backbone tensors + metrics

Server:
10. FedAvg aggregates shared tensors
11. Aggregate weighted client live accuracy
12. Optionally run RF evaluation for this round
13. Save run metrics
14. Save latest checkpoint
15. If best RF so far, preserve best-in-run checkpoint
16. If run beats global best, update best_artifacts/
```

## Metrics Philosophy

The project now intentionally distinguishes between two kinds of performance:

### 1. Client Accuracy

This is the **live training signal**.

It answers:

- Are federated rounds improving the shared representation?
- Are local clients becoming better at classification with their local private heads?

This metric is:

- updated every round
- aggregated from client test accuracy
- the best live signal for round-by-round monitoring

### 2. RF Accuracy

This is the **downstream server-side benchmark**.

It answers:

- How well does the federated backbone support the centralized PCA + RF pipeline?
- How good is the final hybrid representation for the original project objective?

This metric is:

- heavier to compute
- intentionally throttled
- often evaluated only on selected rounds or final round

### Why Two Metrics Matter

The architecture is hybrid:

- clients optimize local neural heads
- server evaluates downstream RF performance

So the two metrics are related but not identical. This is expected and should be communicated clearly to teammates.

## Communication Path

### Original Full-Payload Mindset

Earlier, a full EfficientNet exchange would be roughly:

- about `15.8 MB` per round

### Current Shared-Slice Payload

With partial EfficientNet federation, the payload was reduced to around:

- about `1.6 MB`

This is a major systems-level improvement because it reduces:

- round latency
- gRPC transfer time
- CPU serialization cost
- client-to-server traffic

while keeping EfficientNet-B0 in the system.

## Current Evaluation Strategy

The current design explicitly separates:

- **fast federated training path**
- **heavy downstream evaluation path**

### Fast Path

- skip expensive initial RF evaluation by default
- do not block round 1 on feature extraction over all data
- aggregate client metrics every round

### Heavy Path

- run PCA + RF only on selected rounds
- always run on final round
- persist RF/PCA artifacts for inference or later comparison

This design is one of the most important engineering improvements in the project.

## Artifact Management

### Per-Run Preservation

Each run now preserves:

- metrics
- backbone checkpoints
- PCA artifact
- RF artifact
- run summary

This allows:

- reproducibility
- run comparison
- rollback
- confidence while experimenting

### Best-Run Preservation

The best run across experiments is preserved separately.

Current selection rule:

- highest **RF accuracy** wins

This is appropriate because the historical benchmark and final hybrid pipeline are RF-based.

Best preserved artifacts:

- best backbone checkpoint
- best PCA
- best RF
- best-artifact metadata
- best metrics snapshot

## Current Operational Workflow

### Single Command

```powershell
scripts/run_federated.ps1 -NumClients 3 -NumRounds 10 -Epochs 1 -BatchSize 32 -TrainableBlocks 1 -ProgressiveUnfreezeSchedule "1:1,4:2,8:3"
```

### What Happens Internally

1. Launcher generates a unique run id.
2. Launcher creates run-specific metrics and artifact targets.
3. Server starts.
4. Server waits for required clients.
5. Clients connect and train each round.
6. Metrics are updated continuously.
7. Final RF evaluation runs.
8. Artifacts are preserved per run.
9. Best artifacts are updated only if the run is better.

## Current Strong Experimental Baseline

One strong observed run in the current architecture produced:

- Round 10 client accuracy: `94.9%`
- Round 10 RF accuracy: `93.9%`

with:

- `NumClients = 3`
- `NumRounds = 10`
- `Epochs = 1`
- `BatchSize = 32`
- `TrainableBlocks = 1`
- `ProgressiveUnfreezeSchedule = "1:1,4:2,8:3"`

This is important because it shows the current engineered path is no longer just a latency optimization. It is also competitive in accuracy while being much stronger operationally.

## Architectural Strengths of the Current System

- Keeps Flower as the orchestration layer
- Keeps EfficientNet-B0 instead of replacing it with a weaker model
- Reduces payload drastically through partial communication
- Preserves real federated learning
- Avoids startup stalls from expensive initial RF evaluation
- Separates live monitoring from heavy benchmark evaluation
- Supports full-data coverage under capped per-round training
- Supports progressive unfreezing for better accuracy/latency balance
- Offers one-command execution
- Preserves run history and best checkpoints safely

## Known Tradeoffs

- The architecture still optimizes two related but different objectives:
  - local client neural head accuracy
  - server-side RF accuracy
- RF evaluation remains computationally heavy
- Partial EfficientNet fine-tuning is still an approximation to full end-to-end fine-tuning
- Because the system is hybrid, metric interpretation must be explained carefully to others

## Recommended Talking Points for Team / Mentor Discussion

When explaining the current architecture, emphasize:

1. We did not simply make the system faster by removing learning.
   - We preserved federated learning and made the shared model updates meaningful.

2. We did not replace EfficientNet-B0 with a weak shortcut.
   - We kept the stronger backbone and optimized the communication unit.

3. We separated the training loop from heavy downstream evaluation.
   - This improved startup time and round latency significantly.

4. We improved experimentation maturity.
   - Metrics and artifacts are now preserved per run and across best runs.

5. We improved usability.
   - The system now runs from one command and is easier to demonstrate and maintain.

## Conclusion

The current architecture is a much more mature system than the original version.

It is no longer just a proof-of-concept federated pipeline. It is now a structured experimental platform with:

- a stable training workflow
- lower-latency federated rounds
- partial EfficientNet communication
- progressive unfreezing
- clear monitoring semantics
- best-artifact preservation
- simplified one-command execution

This is the version that should be treated as the current engineering baseline for future hyperparameter tuning and model-quality improvements.
