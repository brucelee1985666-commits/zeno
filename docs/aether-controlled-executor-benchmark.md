# Aether controlled executor benchmark: legacy vs Taskflow

Status: benchmark phase prepared; no benchmark has been executed and no executor has been replaced.

## Non-negotiable safety boundary

- The legacy Zeno executor remains the production/default path.
- Taskflow remains opt-in and isolated behind `ZENO_AETHER_ENABLE_TASKFLOW`.
- No merge into `master`, no default flip, and no automatic dispatch of arbitrary `INode` objects.
- Benchmark-only instrumentation must not alter graph semantics, node order where order is observable, or resource ownership.
- A performance result is invalid if the two backends do not produce equivalent outputs.

## Objective

Measure whether the Taskflow adapter improves scheduling on representative Zeno workloads, while separating scheduler overhead from useful node work. This is an evidence-gathering phase, not an activation decision.

## Workload selection

Start by inventorying and classifying existing Zeno graphs rather than inventing a single synthetic DAG. Candidate fixtures in `misc/graphs/` include:

| Candidate | Workload family to verify from the graph contents | What it stresses |
|---|---|---|
| `flip.zsg` | FLIP / fluid simulation | iterative dependencies, stateful simulation boundaries |
| `rigid.zsg` | rigid-body simulation | many independent bodies plus synchronization points |
| `cloud.zsg` | volumetric/cloud workflow | volume processing and memory pressure |
| `Heightfield Composite.zsg` | heightfield composition | broad procedural graph and data fan-out |
| `Noise_fBm_texture.zsg` | procedural noise | fine-grained CPU work and scheduler overhead |
| `WB_Erode_Test_137_z3.zsg` | terrain erosion | long-running numerical work and iterative stages |

These are candidates, not yet certified benchmark fixtures. Before timing, record the graph hash, node count, edge count, node-type histogram, external assets, frame/iteration settings, and required plugins. Exclude a fixture if it is nondeterministic, depends on unavailable assets, or cannot be run identically by both paths.

## Comparison stages

### Stage 0 — baseline and graph inventory

1. Pin the exact commit SHA, compiler, build type, C++ standard, CPU model, core/thread count, RAM, OS, and relevant GPU/driver details.
2. Preserve a clean legacy-only build and record its existing behavior.
3. Parse candidate `.zsg` files and record topology and execution-relevant settings.
4. Identify the actual legacy entry point and add read-only timing hooks around graph scheduling and node execution. Do not change scheduling behavior.

### Stage 1 — scheduler microbenchmarks

Use deterministic, side-effect-free callback DAGs with topologies derived from measured Zeno graph shapes:
- chain / narrow dependency pipeline;
- wide fan-out and fan-in;
- mixed-width layered DAG;
- many tiny tasks (scheduler-overhead stress);
- fewer coarse tasks (work-dominated case).

Run the same callback payload and dependency edges through the legacy reference harness and Taskflow. Keep data generation outside the timed region. Validate output checksums after every run.

### Stage 2 — representative graph replay

Only replay real graph work where node callbacks can be represented safely and equivalently. Until a node capability/resource contract and immutable execution snapshot exist, do not submit arbitrary live `INode` objects to Taskflow. Stateful, Python, GPU/CUDA, file-I/O, and externally side-effecting nodes must remain on the legacy path unless their ordering and resource ownership are explicitly proven safe.

### Stage 3 — decision gate

A Taskflow pilot may be proposed only if all correctness and stability gates pass and the performance gain is repeatable on more than one representative workload. A microbenchmark win alone is insufficient.

## Measurement protocol

- Build both variants in Release with identical compiler flags and dependency versions.
- Pin CPU affinity where supported; use the same thread budget for both backends.
- Run 3 warm-up iterations, then at least 20 measured iterations per fixture.
- Alternate backend order (ABBA / randomized blocks) to reduce thermal and cache bias.
- Record median, p90, p95, minimum, and coefficient of variation; report raw samples.
- Separate graph preparation, scheduler submission/dispatch, useful node work, synchronization/wait, and total wall time.
- Record tasks/nodes completed, task count, dependency count, worker utilization, context switches if available, peak RSS, GPU transfer bytes/latency, and peak VRAM for GPU-enabled fixtures.
- Run correctness checksums and deterministic output comparisons outside timing.
- Report cold-start and warm steady-state separately.

## Required result table

For every fixture and backend, publish:
- commit SHA and fixture SHA-256;
- compiler/build configuration and hardware;
- median / p90 / p95 total wall time;
- scheduler overhead and useful-work time;
- throughput (nodes or tasks per second);
- peak RAM and VRAM, transfer bytes and latency where applicable;
- output checksum / numerical tolerance result;
- failures, skipped nodes, and run count.

Report speedup as `legacy_median / taskflow_median`. Do not aggregate unlike workloads into a single headline number; show per-fixture results and a geometric mean only as a secondary summary.

## Pass / fail gates

**Correctness (mandatory):**
- identical graph inputs and dependency edges;
- equivalent outputs (exact checksum for deterministic CPU cases; documented numerical tolerance for floating-point cases);
- no missing, duplicated, or reordered side effects;
- exceptions and cancellation are surfaced rather than swallowed;
- no data races under available sanitizer/debug instrumentation.

**Performance (evidence, not an automatic merge):**
- Taskflow must beat the legacy path beyond measurement noise on at least two materially different CPU workloads;
- no severe regression on fine-grained or dependency-heavy graphs;
- scheduler overhead must not erase gains on realistic task sizes;
- memory and thread usage must remain within an explicitly documented budget.

**Stability:**
- 20+ measured iterations per fixture with low enough variance to distinguish the result;
- no hangs, deadlocks, or intermittent checksum failures;
- repeat the winning cases on a second machine/configuration before proposing production use.

## Known blockers

The current adapter accepts explicit callbacks and dependency edges; it is not an `INode` executor. Zeno's recursive input resolution and mutable shared graph/session state (including `Context::visited`) mean arbitrary node-level parallelism is not safe without a capability contract and immutable execution plan. GPU/CUDA ownership remains with existing ZPC/CUDA paths.

## Current conclusion

No scheduler adoption decision is authorized by this benchmark plan. The legacy executor remains authoritative. The next implementation task is to add benchmark instrumentation and a fixture-inventory tool on this isolated branch, then execute the protocol on real hardware. If hardware is unavailable, report the harness as prepared and unexecuted—never fabricate timings or claim a performance win.
