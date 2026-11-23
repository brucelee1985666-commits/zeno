# Aether native-node parallelization candidates

Status: source-level inventory only. No production node is opted into parallel
execution by this document, and no ordinary `INode::apply()` call is routed
through Taskflow.

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

## First candidates from the legacy oldzenbase extension

These are candidates for a follow-up adapter, not approved opt-ins yet.

| Node | Source | Static assessment | Gate before opt-in |
|---|---|---|---|
| `TransformMesh` | `projects/oldzenbase/mesh/TransformMesh.cpp` | Reads a mesh and optional numeric inputs, creates a new mesh, and writes only its own output. No global state is apparent in the inspected body. | Add tests for singular transforms, empty/mismatched mesh attributes, and confirm all input object accessors are read-only. |
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

- No production node currently overrides `getAetherExecutionCapabilities()`;
  the default contract remains conservative.
- `oldzenbase` is optional and categorized as deprecated.
- The Taskflow prepared callback bridge is a prototype, not an integration
  replacement for `Graph::applyNodes`.
- No C++ build, CTest, race sanitizer, runtime graph test, or benchmark result
  is claimed by this inventory.
