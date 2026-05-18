# Improvements Report: From Original Architecture to Current Federated EfficientNet System

## Purpose

This document explains how the project evolved from the original architecture in [Architecture.md](</d:/Tech stuffs/flwr-flood-damage/Architecture.md>) to the current system documented in [New_Architecture.md](</d:/Tech stuffs/flwr-flood-damage/New_Architecture.md>).

The goal is not only to show accuracy differences, but to explain the full engineering improvement across:

- latency
- federated correctness
- communication efficiency
- observability
- experimental safety
- operational simplicity
- reproducibility

This document is intended for:

- team handoff
- mentor explanation
- design review discussions
- project documentation for future iterations

## Executive Summary

The original architecture proved that federated disaster classification was possible, but it behaved more like a research prototype.

The current architecture preserves the federated objective while substantially improving the system in practical engineering terms:

- EfficientNet-B0 is retained for stronger representations.
- The communication unit is reduced from full-model style exchange to partial shared-slice exchange.
- Startup and per-round latency are reduced by removing expensive RF work from the critical path.
- Monitoring is more honest and more useful.
- Running the project is much easier.
- Each run now preserves its own metrics and artifacts.
- The best-performing experiment is protected from accidental overwrite.

This means the system improved not only in model quality, but in **systems quality**.

## Original Architecture Summary

The earlier system had these defining characteristics:

### Strengths

- Flower-based federated setup
- privacy-preserving data locality
- server-side PCA + RandomForest hybrid pipeline
- working dashboard
- multi-client structure
- valid end-to-end demonstration

### Limitations

- server-side RF evaluation was too tightly coupled to the training loop
- initial server evaluation created startup delay
- the communication path was heavier than necessary for EfficientNet
- the system used multiple terminals and more manual orchestration
- metrics from different runs were not preserved safely
- artifact management was not robust for repeated experimentation
- the dashboard mixed evaluation semantics in a way that could appear flat or confusing

## Side-by-Side Comparison

| Area | Original Architecture | Current Architecture | Why Current is Better |
|------|-----------------------|----------------------|-----------------------|
| **Backbone** | SimpleCNN emphasized in design, EfficientNet optional | EfficientNet-B0 is the main engineered path | Better visual feature quality while preserving FL |
| **Federated Unit** | Full shared backbone mindset | Only selected trainable EfficientNet blocks are shared | Lower payload, lower latency, same FL principle |
| **Client Training** | Simpler local training path | Local head + selective backbone fine-tuning | More meaningful global learning |
| **Round Scheduling** | Static training shape | Progressive unfreezing supported | Better accuracy/latency tradeoff |
| **Per-Round Local Data Use** | Full local loader every round | Full coverage across rounds with optional capped batches | Better control of latency without losing dataset coverage |
| **Initial Evaluation** | Expensive initial server work | Initial RF evaluation skipped by default | Faster startup |
| **Server Evaluation** | Heavy RF path closely tied to every round | Throttled RF evaluation, final-round emphasis | Faster rounds, less blocking |
| **Monitoring** | Single accuracy view | Separate client live accuracy and RF accuracy | More honest interpretation |
| **Launcher** | Multiple manual terminals | One-command orchestration | Much better usability |
| **Metrics Persistence** | Single metrics file style | Per-run metrics files + latest pointer + best snapshot | Better experimentation hygiene |
| **Artifact Management** | No robust per-run/best preservation in server flow | Per-run artifacts + protected best artifacts | Prevents regression overwrites |
| **Payload Size** | Full EfficientNet mindset around ~15.8 MB | Partial shared slice around ~1.6 MB | Major communication improvement |
| **Device Logic** | CPU hard bias existed | Auto-selects CUDA/MPS/CPU | Better hardware utilization |
| **Experimental Repeatability** | Weaker | Stronger through run summaries and checkpoints | Easier comparisons and rollback |

## Improvement Area 1: Federated Learning Became More Correct

One subtle but important issue in the older behavior was that the shared representation learning was not always as meaningful as the communication cost suggested.

The current system improves this by making the communicated model slice intentional.

### Before

- there was a risk of paying large communication cost without proportional shared learning benefit
- the shared model design was less aligned with the real bottlenecks

### Now

- the shared slice is explicitly defined
- the server and clients agree on exactly which EfficientNet tensors are federated
- the system still performs real FedAvg over shared parameters

### Why this matters

This makes the FL story stronger:

- data is local
- shared parameters are meaningful
- communication cost is justified by actual global learning

## Improvement Area 2: Communication Efficiency Improved Dramatically

This is one of the biggest practical gains.

### Original Situation

With EfficientNet-B0, a full shared payload was roughly:

- `~15.8 MB`

That is expensive for:

- local CPU systems
- multi-node federated rounds
- repeated Flower serialization
- network transfer

### Current Situation

By federating only the selected trainable EfficientNet slice, the payload dropped to around:

- `~1.6 MB`

### Why this matters

This improves:

- round time
- network efficiency
- CPU overhead
- scalability to more nodes

### Why this is an important engineering improvement

We did **not** get this gain by discarding EfficientNet-B0.  
We got it by changing **what is communicated**, which is the more mature systems solution.

## Improvement Area 3: Startup Latency Improved

The older flow allowed heavy evaluation to block the beginning of training.

### Problem

At server startup, Flower can call evaluation on round 0. If that path immediately runs:

- full feature extraction
- PCA fit
- RF fit
- evaluation over all relevant images

then the system looks frozen before training even begins.

### Improvement

The server now skips expensive initial RF evaluation by default.

### Impact

- training begins much faster
- user confidence improves
- the system feels alive sooner
- demonstration experience is much better

This is a major usability and systems improvement, even though it is not directly an accuracy improvement.

## Improvement Area 4: Per-Round Latency Improved

### Original Issue

Heavy server-side PCA + RF work sat too close to the critical training path.

### Current Fix

RF evaluation is throttled:

- it can be skipped on intermediate rounds
- it still runs when needed, especially on the final round

### Why this matters

This separates:

- **training loop**
- **benchmark loop**

That is a much better engineering pattern because the backbone can keep learning without repeatedly paying the heaviest downstream evaluation cost.

## Improvement Area 5: Better Use of EfficientNet-B0

The key design challenge was:

- keep EfficientNet quality
- avoid full end-to-end cost every round

### Current Solution

- partial block unfreezing
- selective tensor sharing
- progressive unfreezing

### Why progressive unfreezing matters

The schedule such as:

```text
1:1,4:2,8:3
```

means:

- start cheap and stable
- grow capacity later
- pay more compute only when the model has already stabilized somewhat

This is a more productively engineered way to use a large pretrained model under FL constraints.

## Improvement Area 6: Local Data Usage Became More Controlled

Earlier, local training was more all-or-nothing per round.

The current system introduced a better mechanism:

- `max_batches_per_round`
- full-coverage chunking across rounds
- reshuffle only after full dataset pass

### Why this matters

This provides a better balance between:

- latency control
- statistical coverage
- fairness across a client’s private data

It is better than both extremes:

- always full local data every round
- random tiny subsets with no coverage guarantee

## Improvement Area 7: Metrics Became Much More Honest

This is one of the most important conceptual improvements.

### Original Problem

When only RF evaluation was visible, skipped RF rounds could make the curve look flat even if the backbone was actually improving.

### Current Fix

The system now separates:

- **Client Accuracy**
  - live every round
- **RF Accuracy**
  - heavy downstream metric on selected rounds

### Why this matters

This gives teammates and mentors a much clearer story:

- client accuracy tells us if federated training is progressing
- RF accuracy tells us how the downstream hybrid pipeline is performing

That is a major observability improvement.

## Improvement Area 8: The Dashboard Became Operationally Correct

The dashboard now:

- reads the latest run automatically
- knows when training is complete
- counts rounds correctly
- distinguishes live client accuracy from selective RF accuracy
- stops auto-refresh after completion

### Why this matters

A dashboard is not only cosmetic. It is a debugging and communication tool.

When the dashboard is wrong:

- users lose trust
- mentor demos become confusing
- training interpretation becomes harder

The current dashboard is far more aligned with the real runtime behavior.

## Improvement Area 9: Running the System Became Much Easier

### Original Workflow

- one server terminal
- three client terminals
- manual coordination

### Current Workflow

One command:

```powershell
scripts/run_federated.ps1 -NumClients 3 -NumRounds 10 -Epochs 1 -BatchSize 32 -TrainableBlocks 1 -ProgressiveUnfreezeSchedule "1:1,4:2,8:3"
```

### Why this matters

This is a major product-engineering upgrade:

- easier demos
- fewer setup errors
- easier onboarding
- lower cognitive load
- better repeatability

It also better matches how a polished engineering tool should feel to use.

## Improvement Area 10: Experiment Safety Improved Dramatically

This is one of the most mature changes in the system.

### Original Risk

When experimenting repeatedly, weaker runs could overwrite useful outputs.

### Current Design

Each run now preserves its own:

- metrics
- latest backbone
- best-in-run backbone
- final backbone
- PCA
- RF
- run summary

And the system separately preserves global best artifacts in:

- `best_metrics.json`
- `best_artifacts/`

### Why this matters

This protects the project from a very common research-engineering failure mode:

- “We found a good run, then lost the exact model state during later experiments.”

That risk is now substantially reduced.

## Improvement Area 11: Hardware Utilization Improved

The utility layer now uses:

- CUDA if available
- MPS if available
- CPU otherwise

### Why this matters

The previous CPU bias left performance on the table.  
The current device-selection logic makes the system more portable and more realistic for different environments.

## Improvement Area 12: Reproducibility Improved

The current architecture is stronger for reproducible experiments because it now keeps:

- unique run metrics
- run summary
- best metrics snapshot
- best artifact snapshot
- preserved progressive schedule configuration in logs

This makes it far easier to explain:

- which run produced which result
- which checkpoint belongs to which experiment
- why a certain configuration was selected as best

## Why the Current Setup Matters Even Beyond Accuracy

A team or mentor may naturally ask:

“Why is this architecture better if accuracy is similar or only slightly better?”

The answer is:

because the project is now better across the full engineering stack.

### It is better in latency

- lower payload
- skipped heavy startup evaluation
- throttled RF evaluation
- capped local batch option

### It is better in reliability

- cleaner server/client parameter contract
- more stable one-command execution
- better metrics semantics

### It is better in observability

- client and RF signals are separated
- round status is clearer
- dashboard behavior is more correct

### It is better in experimentation

- run-by-run preservation
- best-run preservation
- no accidental overwrite of strongest artifacts

### It is better in maintainability

- simpler to launch
- easier to explain
- easier to compare configurations

## Example of Current Performance Improvement Story

A strong current run reached:

- `client accuracy = 94.9%`
- `RF accuracy = 93.9%`

with:

- EfficientNet-B0
- 10 rounds
- progressive unfreezing
- one-command launcher
- protected artifact preservation

Even when compared with an older peak RF-oriented result around `94.7%`, the current system is still much stronger as an engineering artifact because it now combines:

- strong accuracy
- lower latency
- controlled communication cost
- clearer monitoring
- safer experimentation

This is the right foundation for future hyperparameter tuning.

## Element-by-Element Importance in the Current Setup

### Flower

Why important:

- keeps the system genuinely federated
- handles client-server orchestration
- allows future scaling patterns

### EfficientNet-B0

Why important:

- better visual representation quality than SimpleCNN
- stronger transfer-learning baseline
- helps preserve model quality while optimizing the systems around it

### Partial Block Training

Why important:

- biggest communication reduction without abandoning EfficientNet
- central to the latency improvement story

### Progressive Unfreezing

Why important:

- lets us trade off efficiency and depth over time
- improved current strong results

### Local Private Heads

Why important:

- preserve privacy
- allow local supervised adaptation
- keep clients flexible

### PCA + RandomForest

Why important:

- preserves the original hybrid inference idea
- gives a strong downstream benchmark
- acts as a stable non-neural classifier on top of learned representations

### Client Accuracy Metric

Why important:

- best live signal of federated learning progress
- helps avoid misleading flat dashboards

### RF Accuracy Metric

Why important:

- best metric for the hybrid downstream pipeline
- anchors continuity with earlier architecture benchmarks

### One-Command Launcher

Why important:

- operational simplicity
- easier demos
- fewer user errors

### Run-Specific Artifacts

Why important:

- experiment traceability
- reproducibility
- rollback safety

### Best Artifact Preservation

Why important:

- protects strongest model state
- supports long-term hyperparameter search without regression loss

## Recommended Message to Team Members and Mentor

If you need to explain the improvement succinctly, this is the core message:

> We kept the federated-learning objective intact, kept EfficientNet-B0 for stronger representations, reduced communication cost by federating only the meaningful trainable slice, separated heavy RF evaluation from the fast training path, made monitoring more honest, and made experimentation safer by preserving per-run and best-run artifacts.

That sentence captures the essence of the engineering progress.

## Final Conclusion

The move from the original architecture to the current one is not just a tuning change.

It is a broad engineering upgrade across:

- model communication
- runtime latency
- startup behavior
- metrics quality
- orchestration simplicity
- artifact safety
- reproducibility
- maintainability

The current architecture should therefore be seen as the **new stable baseline** for the project.

Future work can now focus on:

- hyperparameter optimization
- RF tuning
- schedule tuning
- round/epoch budget tuning
- deeper backbone adaptation

because the underlying system foundation is now much stronger.
