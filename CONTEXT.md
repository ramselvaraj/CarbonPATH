# CarbonPATH Evaluation

This context defines the language for evaluating ATLAS computation graphs on
CarbonPATH chiplet architectures. It separates graph execution decisions from
the replaceable models that estimate individual operations.

## Language

**ATLAS graph**:
A versioned graph JSON artifact emitted by ATLAS. It is CarbonPATH's external
workload source, rather than a CarbonPATH-owned normalized workload format.
_Avoid_: parser workload, normalized ATLAS workload

**ATLAS operation**:
One immutable CarbonPATH view of a recognized ATLAS graph node. It provides the
generic execution facts (operation type, input and output tensors, GEMM
dimensions) and a read-only ATLAS source view.
_Avoid_: node, parsed node

**ATLAS source view**:
The immutable, typed access path to the original facts of one ATLAS node. Only
an operation input adapter uses it, to build an evaluator input view.
_Avoid_: raw node, attrs dictionary

**Operation input adapter**:
The evaluator-owned adapter that turns an ATLAS operation and tensor access plan
into the operation-specific input view required by one operation evaluator.
_Avoid_: operation handler, parser

**Operation evaluator**:
A replaceable model that estimates one graph operation at an assigned endpoint
from an operation-specific input view.
_Avoid_: operation handler, operation branch

**Evaluation profile**:
The named selection of operation evaluators, assignment policy, tensor movement
policy, and transfer cost model used for one evaluation.
_Avoid_: global configuration, estimator mode

**Operation assignment**:
The choice of the endpoint or endpoint group that runs one graph operation. It
is distinct from GEMM tile mapping, which works inside an SA group.
_Avoid_: operation placement, placement policy, mapping

**GEMM tile mapping**:
The assignment of the tiles of one GEMM to systolic arrays within its assigned
SA group.
_Avoid_: operation assignment

**Tensor residency**:
The endpoint and storage location at which a produced tensor, or a defined
region of it, is available to a consumer.
_Avoid_: tensor assignment

**Tensor movement plan**:
The selected retention, forwarding, replication, or DRAM-spill path that makes
a tensor residency available to a consumer.
_Avoid_: transfer estimate, boundary policy

**Transfer cost model**:
A replaceable model that estimates latency and dynamic energy for one planned
tensor movement over a resolved path.
_Avoid_: routing policy, operation evaluator

**Movement policy**:
The replaceable adapter that plans one adjacent graph handoff. It exposes one
call shape, `plan(request)`, and owns the movement rules behind that seam. The
named movement policy is a code identifier, not a domain decision.
_Avoid_: boundary policy, movement handler

**Fused activation**:
A ReLU that runs on the systolic arrays of the GEMM directly before it. It is a
workload-mapping choice of one GEMM -> ReLU handoff, shown by the movement
policy `fused_sa_activation_v1`. The tensor does not move, and the GEMM is the
producer of the handoff that leaves the ReLU.
_Avoid_: merged operation, SA ReLU

**Movement seam**:
The single interface between the executor and a movement policy: the executor
supplies a movement request and receives a movement plan, without inspecting
GEMM tiles or branching on operation type.
_Avoid_: movement branch, boundary call site

**Movement phase**:
The movement for one adjacent graph handoff. Its latency is the maximum routed
slice completion; its energy is the sum of every physical slice's energy.
_Avoid_: transfer estimate, boundary cost

**Routed tensor slice**:
One physical transfer of a rectangular tensor region between two endpoints,
with its resolved route, byte count, latency, and energy. Multiple slices make
up one movement phase; replication means physical bytes can exceed logical
bytes.
_Avoid_: transfer slice, tile

**Modular design point**:
One architecture paired with one workload mapping, evaluated by the modular
executor. It is the unit that modular annealing proposes, scores, and accepts or
rejects. The ATLAS graph and the evaluation profile are fixed for the whole
search.
_Avoid_: candidate, configuration

**Workload mapping**:
The part of a design point that says how the workload uses the architecture: the
endpoints of each operation, the dataflow, split-K and tile order of each GEMM,
and the movement policy of each handoff. Annealing changes it. The evaluation
profile gives only its first values.
_Avoid_: profile knob, evaluator setting, placement policy

**Design point file**:
The one result file of a run. It holds the best architecture and its workload
mapping in two sections, with the name and fingerprint of the workload graph
and of the evaluation profile. It holds no scores. A run can start from it.
_Avoid_: best architecture file, best profile file

**One-option parameter**:
A parameter that is in the loop but whose option list in the search space has
one entry, such as the FPGA count or the link timing. The search does not select
its move. A second entry puts it in the search with no code change.
_Avoid_: fixed parameter, hard-coded parameter

**Final-result equivalence**:
For an equivalent sequential-GEMM workload and search definition, modular
annealing returns the same canonical best architecture, numerically equivalent
raw metrics, and numerically equivalent objective value as original CarbonPATH.
It does not require both searches to follow the same proposal or acceptance
trace. The best design changes only on a strict objective improvement, so of
several equal-scoring architectures the search keeps the first one it reached.
_Avoid_: trace equivalence, legacy mode

**ATLAS objective**:
A replaceable model that turns one modular design point's raw metrics (latency,
energy, area, cost, carbon) into the single scalar the search minimizes. Its
configuration may include metric weights, normalization, and calibration data.
_Avoid_: cost function, calibration

**Placeholder policy**:
A deliberately bad movement, transfer, or evaluator model used only to check that
the search penalizes a known bad policy and does not select it. Placeholder
policies are not hardware characterizations and are isolated for deletion.
_Avoid_: baseline, reference policy

**Unsupported evaluation**:
An evaluation that cannot run because its ATLAS graph, evaluation profile, or
architecture uses a capability CarbonPATH does not yet model.
_Avoid_: partial result, infeasible estimate
