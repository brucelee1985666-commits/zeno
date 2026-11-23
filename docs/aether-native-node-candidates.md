# Aether native-node parallelization candidates

Status: first native adapter implemented for `TransformMesh` in the optional
`oldzenbase` extension. It is callable only through the prepared-callback API;
ordinary `INode::apply()` is not routed through Taskflow.

## Safety gate

The capability contract is intentionally conservative. A node may only be
declared CPU-parallel-safe after all of the following are true:

1. It reads inputs without mutating shared input objects.
2. It creates outputs owned by that invocation (or writes to an explicitly
   declared, disjoint resource).
3. It has no global, session, filesystem, UI, random-generator, or hidden
   graph-context side effects.
4. It is deterministic for the same input values.
5. Its implementation and any helper it calls are safe when invoked
   concurrently with other nodes.

The current Taskflow adapter accepts prepared callbacks, not `INode` objects.
It deliberately does not call the protected `apply()` method, `preApply()`,
or `Graph::applyNode()`. A capability declaration alone therefore does not
make a production node execute concurrently. A future native adapter must
supply a reviewed per-node callback and preserve input/output ownership.

## Implemented first adapter: TransformMesh

`projects/oldzenbase/mesh/TransformMesh.cpp` now declares a conservative CPU
capability contract and exposes `isAetherTransformMeshNode()` plus
`makeAetherTransformMeshPreparedWork()`. The callback consumes the prepared
input snapshot, reads the source mesh without modifying it, and returns a new
mesh output. It does not call `INode::apply()` or write to the graph's output
map. The shared transform implementation also rejects singular normal
transforms and non-finite output coordinates. A dedicated smoke test checks
translation, normal handling, output isolation, and singular-scale rejection.

This is an opt-in building block, not an automatic scheduler integration: a
caller must identify the node, provide the callback in the work map, and invoke
the prepared Taskflow runner. Legacy graph execution remains unchanged.

## Remaining candidates from the legacy oldzenbase extension

These are candidates for follow-up adapters, not approved opt-ins yet.

| Node | Source | Static assessment | Gate before opt-in |
|---|---|---|---|
| `GeoVertexVel` | `projects/oldzenbase/mesh/PolyVelocity.cpp` | Reads two meshes and `dt`, creates a new particle object, and writes only its own output. | Validate equal vertex counts and nonzero `dt`; test malformed inputs before enabling. |
| `PrimitiveToParticles` | `projects/oldzenbase/pars/PrimitiveToParticles.cpp` | Reads primitive attributes and creates a new particle object. No external side effects are apparent. | Test absent/malformed `pos` and `vel` attributes and confirm output ownership. |
| `ParticlesToPrimitive` | `projects/oldzenbase/pars/ParticlesToPrimitive.cpp` | Reads particle arrays and creates a new primitive object. No external side effects are apparent. | Test unequal position/velocity lengths and attribute initialization. |

All four are inside the optional `oldzenbase` extension; the extension is OFF
by default. These source-level observations are not a substitute for runtime
race testing or object-lifetime review.

## Excluded from the first batch

- `RandomParticles` uses the process-global `drand48()` generator (or
  `rand()` on MSVC); its random state is not an explicit per-node resource.
- `MeshCopy` mutates the supplied `copyTo` mesh, so separate nodes may alias
  and write the same object.
- `MeshToPrimitive` and `PrimitiveToMesh` have `ConvertTo` subclasses in the
  same source file that call `move_assign` on an input object. Do not put
  a CPU-safe capability on either base class unless the mutating subclasses
  are separately overridden to remain conservative.
- `PolyDuplicate` clones then mutates cloned meshes. The clone deep-copy
  semantics and nested OpenMP behavior need explicit validation first.

## Required next implementation increment

1. Add focused tests for each candidate data contract and aliasing behavior.
2. Add a native callback factory that prepares immutable inputs and returns
   invocation-owned outputs; do not expose or call arbitrary `INode::apply()`
   from worker threads.
3. Add a tiny producer/consumer graph test that runs the native callbacks with
   Taskflow and checks deterministic outputs and no mutation of source objects.
4. Only then opt in the individually reviewed node classes. Keep every other
   node on the legacy executor.

## Current limitations

- `TransformMesh` is the only production node in this extension that currently
overrides `getAetherExecutionCapabilities()` for the Aether CPU contract; all
other nodes retain the conservative default.
- `oldzenbase` is optional and categorized as deprecated.
- The Taskflow prepared callback bridge is a prototype, not an integration
  replacement for `Graph::applyNodes`.
- The TransformMesh smoke test is added to the remote CI matrix but has not yet
  produced a passing run. No successful C++ build, CTest, race sanitizer,
  full runtime graph test, or benchmark result is claimed by this inventory.
