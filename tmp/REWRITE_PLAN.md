# Rewrite plan: persistent self-play workers + in-memory inference memo

> **STATUS: implemented by Opus, 2026-07-25.** All three phases applied. The three open
> items in section 7 were taken at their stated defaults (server is_alive guard IN,
> max_cache_depth=10 / max_entries=100k, cleanup done) since Brian did not veto.
> Tests live in `tmp/test_rewrite.py`. Two deviations from the plan, both noted inline
> below: (a) the `Cache/` rmtree calls in the mains were KEPT, not deleted, because they
> now clean up stale folders left by the old disk cache when upgrading; (b) an unrelated
> latent bug was fixed in passing — the old error paths called `server.terminate()`
> unconditionally, which would have raised NameError instead of the real error whenever
> `use_inference_server` was False.


Planned by Fable 5 on 2026-07-25 for implementation by Opus. Everything here was
verified against the working tree as of this date (post bug-fix session; see
tmp/REPORT.md for that history).

## 0. Problem being solved

Today a worker process lives for exactly one game. `run_self_play` spawns one process
per game (capped at `num_workers` concurrent), each `self_play_task` builds a session,
plays one game, writes to h5, dies, and the parent respawns a replacement with a
recycled worker id. Consequences:

- per GENERATION: ~1000 process spawns, ~1000 session constructions, ~1000 numba
  warm-ups (x2: mcts1 and mcts2 each njit their own `get_terminal_actions`)
- nothing learned by one game survives into the next: `Session_Cache.Cache_Wrapper`
  uses a SQLite-backed diskcache *because* no in-process memory outlives a game
- the parent supervisor busy-waits (`while num_games_left > 0` with no sleep), pinning
  a full core for the whole generation
- the respawn/id-recycling machinery is ~40 lines of the most fragile code in the file

Target: each worker lives for the whole generation, loops over its share of games,
holds one session and one in-memory memo for its lifetime. The disk cache dissolves.

## 1. Invariants that must NOT change

1. `Self_Play` class internals: untouched. It owns one game's lifecycle and is correct.
   A fresh `Self_Play` per game is intended (it owns `self.game` and mutates it).
2. `MCTS.py`: untouched. It type-checks `isinstance(session, Cache_Wrapper)` (line 102)
   and passes `depth` only then — so the wrapper's class NAME and its
   `run(output_names, input_feed, depth)` signature must survive.
3. `Client_Server.py`: untouched (hardening postponed by Brian; audit findings in the
   conversation record — the critical one is "server death = silent worker hang").
4. Worker id == shm slot index, for the worker's entire life. The static split makes
   this trivially true.
5. Resume semantics: `num_games_left` is derived from `game_stats[2]` in the h5, so
   re-running `run_self_play` after any abort tops up the generation. Preserved as-is.
6. h5 writing: one lock, one writer at a time, same dataset naming
   (`boards_N/policies_N/values_N` with consecutive triples per game). Untouched.
7. On worker exitcode != 0: terminate server, unlink shms, raise RuntimeError — same
   fail-fast contract as today.

## 2. File-by-file changes

### 2a. `Session_Cache.py` — rewrite `Cache_Wrapper` internals (Phase 1, independent)

Replace the diskcache implementation entirely. Keep the class name. New shape:

```python
class Cache_Wrapper:
    """In-memory memo of session outputs for the opening region of the game.

    `depth` is the absolute ply of the evaluated state (well, the parent's ply -
    see the comment in MCTS._expand); positions at depth < max_cache_depth are the
    only ones stored, because only the opening region repeats across games. The
    memo lives as long as the worker process, i.e. one generation.
    """
    def __init__(self, session, max_cache_depth=2, max_entries=100_000):
        self.session = session
        self.max_cache_depth = max_cache_depth
        self.max_entries = max_entries          # safety bound, no eviction: opening
        self.cache = {}                          # positions repeat, deep ones don't
        self.hits = 0
        self.misses = 0

    def run(self, output_names, input_feed, depth):
        if depth >= self.max_cache_depth:
            return self.session.run(output_names, input_feed)
        key = input_feed["inputs"].tobytes()
        outputs = self.cache.get(key)
        if outputs is not None:
            self.hits += 1
            return outputs
        self.misses += 1
        outputs = self.session.run(output_names, input_feed)
        if len(self.cache) < self.max_entries:
            self.cache[key] = outputs
        return outputs
```

Notes for the implementer:
- The `path` constructor param is DROPPED (call site updated in 2b). diskcache import
  goes away. `Self_Play.py`'s `__main__` demo block imports diskcache too — delete
  that import and its Cache usage from the demo.
- Drop the `finished_lookup` latch (it existed to limit SQLite traffic; a dict lookup
  is ~100ns). Lookup is gated by the same `depth < max_cache_depth` condition as
  storage — a deep position can never be a hit, so don't hash it.
- `input_feed["inputs"]` at this point is the (1, H, W, C) float32 tensor built in
  `MCTS._compute_outputs`. `.tobytes()` on a non-contiguous view silently makes a
  C-order copy — correct, just noting it's fine.
- Cached outputs are the raw `session.run` return (list of arrays). VERIFIED safe to
  share: `get_legal_actions_policy_MCTS` fancy-indexes the policy (`policy[legal_actions]`
  → copy) before any in-place ops, and dirichlet applies to that copy. No caller
  mutates the cached arrays.
- hits/misses counters exist for the test and for an optional one-line stats print at
  worker exit. Keep them.

### 2b. `Self_Play.py` — `self_play_task` gets a loop (Phase 2)

New signature (two appended params):

```python
def self_play_task(worker_id, info, game_class, input_feed_info, output_feed_info,
                   build_config, train_config, lock, folder_path, generation,
                   num_games, progress):
```

Body: everything up to and including session construction is UNCHANGED and runs once.
The Cache_Wrapper call site becomes:

```python
        if train_config.get("max_cache_depth", 0) not in [None, 0]:
            from Session_Cache import Cache_Wrapper
            session = Cache_Wrapper(session, train_config["max_cache_depth"])
```

Then:

```python
    for _ in range(num_games):
        Self_Play(game_class(), session, build_config, train_config,
                  lock, folder_path, generation).play()
        with progress.get_lock():
            progress.value += 1

    if train_config["use_inference_server"]:
        info.close()
```

- `np.random.seed()` stays where it is: once per process, BEFORE the loop. Never
  per game (a fixed reseed would correlate workers; per-game reseed is pointless).
- Because both `mcts1` and `mcts2` receive the same `session` object and every game's
  `Self_Play` receives it too, wrapping once here gives one shared memo per worker
  for the whole generation. That is the design, not an accident — leave a comment.
- Optional (recommended): after the loop, if the session is a Cache_Wrapper, print
  one line: worker id, hits, misses, len(cache). Cheap observability for tuning
  `max_cache_depth`.

### 2c. `Self_Play.py` — `run_self_play` supervisor (Phase 2)

Everything up to and including server start is unchanged. Replace the spawn loop and
the entire respawn/poll block (currently `lock = mp.Lock()` down to the final joins)
with:

```python
    lock = mp.Lock()
    progress = mp.Value("i", 0)

    shares = [num_games_left // num_workers + (i < num_games_left % num_workers)
              for i in range(num_workers)]

    jobs = []
    for worker_id in range(num_workers):
        worker = mp.Process(target=self_play_task,
                            args=(worker_id,
                                  shms[worker_id] if train_config["use_inference_server"]
                                  else [providers, onnx_file_path],
                                  game_class,
                                  convert_to_single_info(batched_input_feed_info),
                                  convert_to_single_info(batched_output_feed_info),
                                  build_config, train_config, lock, folder_path,
                                  generation, shares[worker_id], progress),
                            name=f"{worker_id}")
        worker.start()
        jobs.append(worker)

    last = 0
    while any(w.is_alive() for w in jobs):
        time.sleep(0.25)
        current = progress.value
        if current > last:
            bar.update(current - last)
            last = current
        if train_config["use_inference_server"] and not server.is_alive():
            for w in jobs:
                w.terminate()
            for shm in shms:
                shm.unlink()
            raise RuntimeError("Inference server died mid-generation; self play cannot continue")
    bar.update(progress.value - last)
    bar.close()

    for worker in jobs:
        worker.join()
        if worker.exitcode != 0:
            if train_config["use_inference_server"]:
                server.terminate()
            for shm in shms:
                shm.unlink()
            raise RuntimeError("Main process stopped as self play worker ran into an error")

    if train_config["use_inference_server"]:
        server.terminate()
        for shm in shms:
            shm.unlink()

    with h5.File(f"{folder_path}/Self_Play_Data.h5", "r") as file:
        played = int(file["game_stats"][2])
    if played < train_config["games_per_generation"]:
        warn(f"Generation ended {train_config['games_per_generation'] - played} games short; "
             f"re-running this generation will top it up.")
```

Notes:
- The `time.sleep(0.25)` replaces today's busy-wait — the parent currently pins a
  full core all generation.
- The `server.is_alive()` guard is 4 lines of the postponed Client_Server audit that
  belong in this loop because we are rewriting this loop anyway. Without it, a dead
  server means all workers spin forever and `is_alive()` stays True for all of them.
  This does NOT touch Client_Server.py. (Brian may veto; default in.)
- `num_workers = min(num_workers, num_games_left)` already happens above — keep, so
  no zero-share workers are spawned.
- `warn` needs `from warnings import warn` (check imports; `h5` and `time` are
  already imported).
- The existing `if worker.exitcode != 0` teardown inside the OLD poll loop is deleted
  along with the loop; the post-join check above is the single failure path now. A
  worker crash mid-share aborts the generation (fail fast), and invariant 5 makes
  re-running resume correctly. This matches today's contract; the only delta is
  blast radius (a crash loses that worker's remaining share until re-run).
- tqdm `bar` creation stays where it is (`total=num_games_left`).

### 2d. Config: `Connect4/Connect4.py` (Phase 3, one line + comment)

`"max_cache_depth": 1` → `"max_cache_depth": 10`.

Rationale to put in the comment: depth here is the absolute ply of the evaluated
position, and only plies < depth are memoised. At 1, only children of the empty board
ever got cached (~8 positions per generation, which is why the disk cache was nearly
a no-op). At 10 the memo covers the opening region, which is the only part that
repeats across games. RAM bound: ~850 bytes/entry (672-byte key + outputs + dict
overhead), capped at 100k entries ≈ 85 MB/worker worst case; realistic opening
dedup keeps it far below that.

Gomoku config: leave alone (already 3; Gomoku is out of scope per Brian).

### 2e. Cleanup (Phase 3, optional but tidy)

- `Connect4/main.py` and `Gomoku/main.py` each rmtree `Grok_Zero_Train/{gen}/Cache`
  before and after `run_self_play`. The folder will no longer be created. The calls
  are harmless no-ops (guarded by `os.path.exists`) — EITHER delete them or leave
  them; deleting is tidier, leaving is zero-risk. Implementer's choice, note it in
  the commit message.
- `Self_Play.py` `__main__` block: remove `from diskcache import Cache` and the
  `cache = Cache(...)` lines (dead after 2a). The rest of the demo can stay.
- requirements: `diskcache` can come out of any requirements file that lists it
  (check `requirements*.txt`; do not touch the env itself).

## 3. What explicitly does NOT change

- `Self_Play` class (init/play/h5 writing)
- `MCTS.py`, `MCTS_Gumbel.py`
- `Client_Server.py` (audit fixes postponed; only the supervisor-side is_alive guard
  from 2c interacts with it, without modifying it)
- `Dataloader.py`, `Train.py`, model builders, optimizers
- The `use_inference_server=False` path's structure (each worker builds its own
  onnxruntime session — unchanged behavior, now amortised over the whole share)
- Gomoku/TicTacToe configs

## 4. Test plan (all runnable headless in WSL_tensorflow_general, no GPU needed)

Pattern for stubbing onnxruntime + running the real Server/Parallelized_Session
already exists in `tmp/audit_client_server.py` — reuse it.

1. **Memo unit test** (Phase 1): construct `Cache_Wrapper` around a fake session
   returning a deterministic function of the input. Assert: (a) same input bytes at
   depth < D → 1 miss then hits, identical outputs; (b) depth >= D bypasses (hits
   + misses unchanged); (c) `max_entries` respected; (d) outputs not mutated by a
   simulated caller doing `policy[legal_actions] /= sum`.
2. **Generation-0 end-to-end** (Phase 2): tiny Connect4 run, `NUMBA_DISABLE_JIT=1`,
   `games_per_generation=12`, `num_workers=3`, `use_gpu=False`,
   `use_inference_server=False`, generation 0 (session=None path). Assert
   `game_stats[2] == 12`, all worker exitcodes 0, no leftover processes.
   (Make a scratch folder with a fresh h5 via `Make_Dataset_File`-equivalent; do NOT
   touch the real training folders.)
3. **Inference-server end-to-end** (Phase 2): same tiny run but generation > 0 with
   the stubbed onnxruntime (fake model outputs), `use_inference_server=True`,
   real `create_shared_memory`/`start_server`/`Parallelized_Session` wiring, memo on
   (`max_cache_depth=10`). Assert game count, clean exit, server terminated, shms
   unlinked, and worker stats line shows hits > 0 by the later games.
4. **Resume test**: run 2 but kill one worker mid-generation (or just run with
   `games_per_generation` larger than delivered), re-invoke `run_self_play`, assert
   it tops up to the full count. (This exercises invariant 5 plus the shortfall warn.)
5. **Timing**: Brian runs one real generation before/after on the 2060 and compares
   wall clock. No automated assertion — report the two numbers.

## 5. Acceptance criteria

- All four headless tests pass.
- `grep -c "mp.Process(target=self_play_task" Self_Play.py` == 1 (respawn machinery
  gone).
- No `diskcache` import anywhere outside Time_Parallel(Deprecated)/.
- A real generation produces the same h5 schema as before (spot-check keys and
  game_stats) and `main.py`'s downstream training consumes it unchanged.
- Parent process CPU during self-play is near-idle (observable in Task Manager;
  previously one full core).

## 6. Risks and mitigations

- **Windows spawn pickling**: `mp.Value` and `mp.Lock` in Process args are standard
  and picklable-by-inheritance on spawn; ints are trivially fine. Risk: low.
- **A worker crash now loses its whole remaining share** (was: one game). Mitigated
  by fail-fast + h5-driven resume (invariant 5) + shortfall warning. Accepted in the
  earlier design discussion.
- **Memo staleness across generations**: impossible by construction — the memo dies
  with the worker process, and workers live exactly one generation.
- **Memo RAM**: bounded by `max_entries` (85 MB/worker worst case at defaults);
  realistic opening dedup keeps it much lower. Stats line makes it observable.
- **Uneven shares → stragglers**: shares differ by at most 1 game; game-length
  variance dominates. Not worth work-stealing complexity now. If it ever matters,
  an mp.Queue of game slots is the upgrade path — do not build it preemptively.

## 6b. Decision: the memo is NOT persisted across restarts (asked by Brian)

The memo maps position -> (policy, value) as judged by ONE network. Across generations
the network retrains, so every entry becomes a stale opinion — the memo dying with the
worker process is the invalidation mechanism, not a limitation. The only boundary where
reloading would be valid is a mid-generation resume (same model.onnx), and: (a) today's
disk cache never survived that either — main.py rmtree's Cache/ before every
run_self_play call; (b) the memo self-warms in the first couple of games because it only
covers the opening region, which is the hottest; (c) doing it properly needs model-hash
keying and corrupt-file handling to avoid silently poisoning evaluations. Cost exceeds
the benefit (seconds, on rare resumes). If profiling ever says otherwise: one pickle per
worker, keyed by a hash of model.onnx, load-if-valid — do not key by generation number.

## 6c. Measured: disk cache vs dict (`tmp/bench_cache.py`)

Connect4-sized keys (672 B), 20k ops over 20k distinct keys, zipf access (hot opening
head + long tail), 65.7% of ops are repeats.

```
                                     per op      vs one 300us inference
new: in-memory dict                   1.5 us              0.5%
old: diskcache (native linux fs)    105.3 us             35.1%
old: diskcache (/mnt/c, real path) 4706.9 us           1569.0%
8 workers: dicts 0.23s wall  vs  shared diskcache 4.16s wall (18x)
```

70x faster on a native fs, 3138x on /mnt/c (where the project lives, so sqlite was
writing through WSL2's 9p bridge). On /mnt/c a cache HIT cost 15.7x the inference it
avoided - net negative.

But the old cache was not slowing training down, it was INERT: the finished_lookup latch
disabled lookups permanently after the first miss (0/5000 hits cold), and
max_cache_depth=1 meant only ~8 positions were ever stored, against ~21,600 NN evals per
game. Key consequence: raising max_cache_depth on the OLD implementation would have made
training dramatically slower, not faster. The dict rewrite is what makes a deep cache
viable at all - the two changes only work together.

Caveat: the comparison is asymmetric on sharing. 8 workers shared one sqlite db; dicts
are per process. Cross-worker sharing is the one real advantage given up. The 28-31%
per-worker hit rate measured after only 3 games (tmp/test_rewrite.py) says the loss is
small, but it is a trade, not a free win.

## 7. Open items for Brian

1. The `server.is_alive()` guard in 2c — in by default, veto if you want the rewrite
   completely orthogonal to the postponed Client_Server work.
2. `max_cache_depth=10` and `max_entries=100_000` defaults — both are judgment calls,
   tunable later via the stats line.
3. 2e cleanup (rmtree lines / diskcache removal from requirements) — do now or leave.
