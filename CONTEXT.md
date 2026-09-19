# CarbonPATH Evaluation

This context defines the language for evaluating ATLAS computation graphs on
CarbonPATH chiplet architectures. It separates graph execution decisions from
the replaceable models that estimate individual operations.

## Language

**ATLAS graph**:
A versioned graph JSON artifact emitted by ATLAS. It is CarbonPATH's external
workload source, rather than a CarbonPATH-owned normalized workload format.
_Avoid_: parser workload, normalized ATLAS workload

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
