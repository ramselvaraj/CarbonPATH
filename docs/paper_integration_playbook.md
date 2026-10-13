# Paper Integration Playbook

Use this playbook when reading a paper to determine which of its components can
be used by modular CarbonPATH. Record the resulting concise paper entry in:

```text
/mnt/windows/Users/ramse/syncthing/obsdian/thesis/PATH AI REPO/ATLAS Evaluators.md
```

This is an integration analysis, not a paper summary. Do not propose changes to
ATLAS graph ingestion or executor flow unless that is explicitly the task.

## CarbonPATH Seams

Assess every reusable paper component against each seam.

| Seam | CarbonPATH responsibility |
|---|---|
| Operation input adapter | Builds one evaluator or policy's typed model input from an ATLAS operation and tensor access plan. |
| Operation evaluator | Estimates one operation at an assigned endpoint; returns compute latency and dynamic energy. |
| Operation assignment policy | Selects an endpoint or endpoint group for an operation. |
| Tensor movement policy | Selects retention, forwarding, replication, or DRAM-spill paths for a tensor residency. |
| Transfer cost model | Estimates latency and dynamic energy for one resolved tensor movement path. |
| Architecture/technology characterization | Supplies hardware capacities, bandwidths, action costs, area, or technology parameters used by models. |
| Outer search/optimization | Generates or ranks candidate architectures, mappings, assignments, or policies. |

## Classification

Every component-to-seam claim uses exactly one classification.

| Status | Meaning |
|---|---|
| `direct` | The component can be called or translated into the seam without changing its model purpose. |
| `adapted` | The component is useful, but CarbonPATH needs a wrapper, translation, or constrained use. |
| `reference-only` | The paper provides a modeling idea or validation target, but no reusable component. |
| `not-applicable` | The paper does not provide a useful component for this seam. |

## Required Record Per Reusable Component

For every `direct`, `adapted`, or `reference-only` claim, record:

| Field | Required content |
|---|---|
| Paper component | Named algorithm, model, representation, cost equation, search method, or characterization source. |
| CarbonPATH seam | One seam from the table above. |
| Status | One classification from the table above. |
| Inputs | Exact CarbonPATH facts required: operation dimensions, access plan, endpoint architecture, tensor residency, resolved route, or candidate set. |
| Outputs | Exact result supplied: latency, energy, action counts, access counts, traffic, route score, selected endpoint, selected movement plan, or ranked candidate. |
| Boundary | What remains owned by CarbonPATH. |
| Evidence | Paper section/page, or `summary-only` when the full source was not checked. |

## Analysis Procedure

1. Identify the paper's concrete components. Do not label the whole paper as an evaluator or policy.
2. For each component, test all seven seams.
3. Record only non-`not-applicable` claims in the paper entry.
4. State the required inputs and produced outputs precisely.
5. State the CarbonPATH boundary to prevent duplicated ownership or double charging.
6. Attach source evidence to every claim.

## Seam Tests

### Operation Input Adapter

Use when the paper defines a workload representation or extracts model-specific
facts.

```text
Inputs: AtlasOperation facts + tensor access plan
Outputs: typed evaluator or policy input
```

### Operation Evaluator

Use when the paper estimates one operation at an already selected endpoint.

```text
Inputs: typed operation facts + endpoint architecture
Outputs: compute_latency_ns + dynamic_energy_pj
```

### Operation Assignment Policy

Use when the paper can score or select candidate endpoints for an operation.

```text
Inputs: operation facts + candidate endpoints
Outputs: selected endpoint or endpoint-group score
```

### Tensor Movement Policy

Use when the paper can select a tensor residency path based on reuse, capacity,
or traffic.

```text
Inputs: produced tensor + candidate residencies + capacities
Outputs: selected retention, forwarding, replication, or spill plan
```

### Transfer Cost Model

Use when the paper estimates a resolved tensor route.

```text
Inputs: tensor bytes + resolved route + link/network parameters
Outputs: transfer latency + dynamic energy
```

### Architecture/Technology Characterization

Use when the paper supplies calibrated costs or constraints consumed by another
CarbonPATH model.

```text
Inputs: architecture or technology attributes
Outputs: capacities, bandwidths, action costs, area, or operating limits
```

### Outer Search/Optimization

Use when the paper searches candidate configurations rather than estimating one
already selected configuration.

```text
Inputs: candidate space + objective + model estimates
Outputs: selected or ranked candidates
```

## `ATLAS Evaluators.md` Entry Format

Use a hybrid entry: one compact table followed only by essential integration
notes.

```md
## <Paper>

| Paper component | CarbonPATH seam | Status | Inputs | Outputs | Boundary | Evidence |
|---|---|---|---|---|---|---|
| <component> | <seam> | <status> | <required facts> | <result> | <CarbonPATH-owned work> | <section/page> |

### Integration Notes

- <Only implementation constraints, unsupported cases, or no-double-counting rules.>
```

Do not repeat the paper abstract, results, or general background in this file.

## Timeloop Example

| Paper component | CarbonPATH seam | Status | Inputs | Outputs | Boundary | Evidence |
|---|---|---|---|---|---|---|
| Workload and mapping representation | Operation input adapter | adapted | Operation dimensions, SA architecture, selected mapping, access plan | Typed Timeloop workload and mapping | Adapter selects only model facts; graph execution remains outside Timeloop | Sec. III-V |
| Tile projection and access counting | Operation evaluator | direct | Timeloop workload, architecture, mapping | Tensor footprints, accesses, reuse, multicast, forwarding, reductions | Evaluator converts final cost to `OperationEstimate` | Sec. VI-A/B |
| Analytical performance model | Operation evaluator | adapted | Access counts, compute throughput, memory/network bandwidth | Compute latency | CarbonPATH owns incoming tensor movement and total aggregation | Sec. VI-C |
| Energy action counts | Architecture/technology characterization | adapted | Access/action counts and component energy costs | Dynamic energy or Accelergy inputs | CarbonPATH owns evaluator result formatting | Sec. VI-D |
| Mapper | Operation assignment policy | adapted | Operation facts, candidate endpoint architectures, objective | Candidate endpoint cost or selected endpoint | CarbonPATH policy owns final endpoint selection | Sec. VII |
| Tile footprint and bypass analysis | Tensor movement policy | adapted | Tensor shape, candidate residencies, capacity, mapping | Retention or spill score | CarbonPATH policy owns the final tensor movement plan | Sec. VI-A |
| NoC action counts | Transfer cost model | adapted | Tensor volume, resolved route, network parameters | Route-local traffic/cost facts | CarbonPATH transfer model owns resolved inter-endpoint route cost | Sec. VI-B/D |

Timeloop can be used in several seams, but each use is a separate CarbonPATH
model integration. Do not assume that an operation evaluator also selects
assignment, movement, or transfer policy.
