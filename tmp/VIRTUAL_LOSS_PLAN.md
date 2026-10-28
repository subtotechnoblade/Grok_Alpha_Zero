# Virtual loss and batched inference: plan

Builds on requirements R1-R14 (agreed earlier). K = leaves in flight per worker.

## Principles

1. Batch-native end to end: MCTS -> session -> server -> engine all move batches. K=1 is a batch of one, not a special path.
2. Each layer owns one concern. MCTS decides which leaves. The server combines requests and routes rows back. The engine owns shapes (padding/buckets). Nobody else pads.
3. The hot path allocates nothing: preallocated batch buffers, rows written in place.
4. K=1 is bit-identical to today (R1). It is the regression anchor for every stage.
5. Every stage ends in a test gate and, where it touches speed, a benchmark gate.

## MCTS design

- In-flight counts live in a separate per-node array (`child_inflight`), lazily allocated and freed when a node has nothing in flight. Real `child_visits`/`child_values` are never touched by VL, so K=1 adds exact zeros (R1).
- The PUCT kernel takes the in-flight array: N_eff = N + n, W_eff = W - vl_value * n, parent N_eff includes in-flight. FPU's parent mean uses the effective sums.
- `_expand` splits into `reserve` (CPU half: do_action, legal actions, terminal check, input state) and `complete` (policy -> priors, sort, fill node).
- Reservation appends a pending child to `node.children` immediately, so `len(children)` advances and the next descent naturally takes the next prior-ordered slot. No duplicate (parent, action) by construction (R5), progressive order intact (R7).
- VL is applied after a leaf is confirmed (one walk up parent pointers), reverted in the same walk as the real backup.
- Terminal leaves back up immediately, never take a slot (R6).

### Duplicates after VL

| Kind | What happens | Handling |
|---|---|---|
| Same (parent, action) twice | Impossible: the slot is taken by the pending child at reservation | none needed |
| Same leaf twice (descent reaches a pending child) | A collision | Abandon the descent (nothing applied yet), count it, end the batch early when collisions hit a cap. Never back up the same value twice (that would be a visit the sequential search would not make) |
| Same position, different nodes (transposition) | Two pending leaves with identical inputs | Accepted: evaluated twice. Rare, and dedup is not worth the complexity (Brian's call) |

## Server design

- The wire format is its own module (`Transport.py`): a client/server channel described only by named arrays
  with a per-row shape and dtype, a max rows per request, and a shared-memory layout derived from that spec.
  It knows nothing about MCTS or onnx. Tested on its own and extensively (stage 5a) before the server uses it.
- One shared-memory slot per worker: header [flag, count], input region K rows, separate output region K rows.
- Each array travels in its own dtype (inputs as int8 = 4x less than float32); the server casts once, vectorised, into a preallocated float32 batch.
- Greedy collection: run whatever requests are ready; no fixed wait. Requests that arrive during a run form the next batch.
- Partial batches are not padded by the server. The engine decides: TensorRT dynamic profile runs any size 1..workers*K; DirectML's BucketedEngine pads to its bucket and trims back (exists today).
- Routing: while gathering, record (worker, row_start, count). After the run, each worker gets outputs[row_start : row_start+count] in one contiguous copy, then its flag. Rows stay contiguous and ordered, so no per-row tags.

## Stages (each independently testable)

0. Harness + baselines. Deterministic fake network, frozen reference root stats for fixed positions (Connect4, Gomoku, gen 0, Dirichlet, tau 0/1, tree reuse, forced win/block, cache). Re-measure sims/s: one process with a local TRT session at batch 1, and 10 workers through the server.
1. In-flight arrays + kernel, still K=1. Gate: R1 bit-identity, R3 apply/revert identity, PUCT drops on a VL'd path.
2. reserve/complete split, still K=1. Gate: R1 again, R6, R7.
3. Batched gather in `run()` with a local session (rt.InferenceSession, None/gen 0, Cache_Wrapper). Exact sim count, collision cap, stats (mean batch, collisions). Gate: R1 at K=1, R2, R4, R5, R8. Benchmark: one process, local TRT, K = 1/4/8/16/32.
4. Strength check (R12): default K vs K=1 at equal sims, >=400 games, colours swapped, policy-target KL. Pick default K and vl_value (1.0 loss vs 0.0 visit-only). Pipeline stopped while it runs.
5a. `Transport.py` alone. Gate: round trips for many specs (dtypes int8/float16/float32/int64, scalar
    and multi-dim rows, 1..max rows), every byte of a slot checked, oversize/wrong-dtype/wrong-shape rejected,
    concurrent client processes with random counts and payloads, no stale data across reuse, graceful close.
5b. Server protocol rewrite on top of it (R9, R10). Gate: fake clients in processes sending random counts 1..K; outputs equal a direct session.run on the concatenation (exact on CPU EP, allclose on TRT fp16); K=1 equals today. TRT profile max and DirectML buckets = workers*K.
6. Integration: Config params `leaves_per_batch` (default 1), `virtual_loss` (vl_value), `max_collisions`; Self_Play wires K to MCTS and server sizing. Benchmark R11 (10 workers, TRT): sims/s, GPU util, real server batch, CPU per worker. Ctrl+C still graceful.
7. Only if stage 6 measurements call for it: two half-batches in flight per worker (hides the round trip), or a numba array tree if CPU-bound.

Gumbel MCTS untouched (R14).
