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
The named selection of operation evaluators, placement policy, and tensor
movement policy used for one evaluation.
_Avoid_: global configuration, estimator mode

**Operation placement**:
The assignment of a graph operation to an endpoint or endpoint group. It is
distinct from the assignment of GEMM tiles within an SA group.
_Avoid_: mapping

**GEMM tile mapping**:
The assignment of the tiles of one GEMM to systolic arrays within its assigned
SA group.
_Avoid_: operation placement

**Tensor residency**:
The endpoint and storage location at which a produced tensor, or a defined
region of it, is available to a consumer.
_Avoid_: tensor placement

**Tensor movement plan**:
The selected retention, forwarding, replication, or DRAM-spill path that makes
a tensor residency available to a consumer.
_Avoid_: transfer estimate, boundary policy

**Transfer cost model**:
A replaceable model that estimates latency and dynamic energy for one planned
tensor movement over a resolved path.
_Avoid_: routing policy, operation evaluator

**Unsupported evaluation**:
An evaluation that cannot run because its ATLAS graph, evaluation profile, or
architecture uses a capability CarbonPATH does not yet model.
_Avoid_: partial result, infeasible estimate
