# 0004. Physical links are undirected edges

- Status: Accepted
- Date: 2026-10-02
- Branch: `modular-carbonpath`

## Context

Distributed SA-group/FPGA movement prices every routed tensor slice over the
resolved package topology. Two slices that share a physical connection must
serialize, so the movement model needs a stable notion of "one physical link"
and a rule for opposite-direction traffic on that link.

The existing topology already stores links as undirected edges:
`ChipletSystem._interconnect_setup` records `links` as
`tuple(sorted((src_id, dst_id)))`, and `Scheduler.__form_links` forms the same
key for its split-K congestion model. The movement seam, however, only had
path-aggregate `get_shortest_path` costs, with no per-link accessor.

## Decision

A physical link is the undirected edge `tuple(sorted((a, b)))`. Opposite-
direction traffic on one edge serializes; the model does not assume full-duplex
links.

`ChipletSystem.link_cost(a, b)` returns one directed physical link's
`(reciprocal bandwidth in ns/byte, energy in pJ/bit)`. A missing edge raises
`UnsupportedEvaluation` rather than returning a zero or `None` cost.

For a movement phase, every slice's load is added to each undirected edge on its
route. A slice's completion estimate is its fixed route latency plus the summed
loads on its route; the phase latency is the maximum slice completion and the
phase energy is the sum of every slice's energy.

## Consequences

- The movement model agrees with the existing split-K congestion accounting in
  `Scheduler`, so the two do not invent competing link identities.
- Bidirectional traffic on one edge is charged as if serialized, which is
  conservative but consistent and easy to audit.
- `link_cost` becomes the deep accessor that hides `interconnect_dict` internals
  from movement policies; it is a small addition to `ChipletSystem`.
- Routes themselves remain shortest-path by bandwidth reciprocal; link identity
  only governs contention, not route selection.
