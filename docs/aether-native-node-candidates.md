# Aether native-node parallelization candidates

Status: four native prepared-callback adapters are implemented in the optional
`oldzenbase` extension: `TransformMesh`, `GeoVertexVel`,
`PrimitiveToParticles`, and `ParticlesToPrimitive`.

They are callable only through the audited prepared-callback API. Ordinary
`INode::apply()` and `Graph::applyNodes` are not redirected to Taskflow.

## Safety gate

A node may be declared CPU-parallel-safe only when:

1. It reads inputs without mutating shared input objects.
2. It creates outputs owned by that invocation.
3. It has no global, session, filesystem, UI, random-generator, or hidden
   graph-context side effects.
4. It is deterministic for identical inputs.
5. Its implementation and helpers are safe under concurrent invocation.

The Taskflow adapter accepts reviewed callbacks, not arbitrary `INode`
instances. It does not call `INode::apply()`, `preApply()`, or
`Graph::applyNode()`. A capability declaration alone never authorizes
execution through Taskflow.

## Implemented adapters

| Node | Contract and validation |
|---|---|
| `TransformMesh` | Reads a source mesh, creates a new mesh, transforms vertices and normals, rejects non-finite results and singular normal transforms. |
| `GeoVertexVel` | Reads two meshes and `dt`, requires matching vertex counts and finite non-zero `dt`, creates a new particle object, rejects non-finite velocities. |
| `PrimitiveToParticles` | Reads primitive positions and optional velocity; validates velocity type/count, initializes missing velocity to zero, creates a new particle object. |
| `ParticlesToPrimitive` | Requires matching position/velocity counts, allocates matching output storage, writes every element, and rejects mismatched arrays. |

Each adapter has a dedicated smoke test registered in the optional Taskflow
CI target. Tests check representative output values, rejection paths, and
that source objects / graph outputs are not mutated by the prepared callback.

The dispatcher `makeAetherOldzenbasePreparedWorkForPlan()` maps only the
reviewed node types to callbacks. It rejects incomplete or unsafe plans,
missing nodes, and unknown node types; it never falls back to arbitrary
`INode::apply()`. The callback runner preserves dependency edges and returns
invocation-owned output maps.

## Explicit exclusions

- `RandomParticles`: uses process-global `drand48()` (or `rand()` on
  MSVC); random state is not an explicit per-node resource.
- `MeshCopy`: mutates the supplied `copyTo` mesh; separate nodes may alias
  and write the same object.
- `MeshToPrimitive` and `PrimitiveToMesh`: `ConvertTo` subclasses in the
  same source file call `move_assign` on an input object. Keep the base class
  conservative until subclasses have separate contracts.
- `PolyDuplicate`: clone/deep-copy semantics and nested OpenMP behavior need
  explicit validation.

## Remaining engineering gates

1. Obtain a successful remote CI run for the default-off core build and the
   Taskflow 3.11 configure/build/CTest matrix. The registered dispatcher
   integration test covers producer/consumer dependencies, fan-out consumers,
   muted outputs, graph-output isolation, and stale-plan rejection.
2. The workflow now includes a ThreadSanitizer job for the dispatcher
   integration test. Its graph has two concurrent consumers and repeats the
   prepared execution 32 times. A successful sanitizer run is still required;
   this focused test cannot prove race freedom for all Zeno graphs or callbacks.
3. Static aliasing review: the four reviewed adapters read their input
   objects and allocate invocation-owned output objects. The prepared snapshot
   itself is shallow: `zany` values may retain shared pointers. Therefore the
   executor does not make arbitrary inputs immutable; safety depends on the
   reviewed callbacks treating those objects as read-only. The fan-out smoke
   test verifies that the current two consumers do not mutate their shared
   particle input. Broader alias/lifetime coverage remains a release gate.
4. Benchmark representative graphs against the legacy executor before
   enabling automatic scheduling. Record graph shape, node count, input sizes,
   CPU/thread configuration, warm-up count, median and p95 wall time, and
   output equivalence. Do not infer speedup from the CPU smoke tests. There is
   currently no automatic routing and no performance claim.

## Current limitations

- Only the four nodes listed above override the Aether CPU capability contract
  in `oldzenbase`; other nodes remain conservative by default.
- `oldzenbase` is optional and deprecated.
- The Taskflow prepared-callback bridge is an opt-in prototype, not a
  replacement for `Graph::applyNodes`.
- The CPU CI does not validate CUDA/OptiX, editor/Qt integration, race
  sanitizer behavior, production runtime semantics, or performance.
