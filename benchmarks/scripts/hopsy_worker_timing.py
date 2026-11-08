"""Opt-in timers around unchanged hopsy C++draw calls; no installed edits."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import hopsy.misc as _misc

_s,_c = _misc._s,_misc._c
PROOF = {}

def _emit(chain_idx,n_samples,thinning,draw_wall_ns,draw_cpu_ns,draw_calls,entry_ns):
    directory = Path(os.environ["BENCH_HOPSY_WORKER_TIMING_DIR"])
    result = dict(pid=os.getpid(),chain_idx=chain_idx,n_samples=n_samples,thinning=thinning,
                  draw_calls=draw_calls,draw_call_wall_seconds=draw_wall_ns/1e9,
                  draw_call_thread_cpu_seconds=draw_cpu_ns/1e9,
                  worker_body_elapsed_before_return_seconds=(time.perf_counter_ns()-entry_ns)/1e9,
                  scope="Only MarkovChain.draw call intervals contribute to draw timers. Includes C++sampling and Python binding/state return copy. Excludes Python state-list bookkeeping, final array assembly, pickling, IPC, startup and log writing.")
    target = directory / f"pid{os.getpid()}-chain{chain_idx}-{time.time_ns()}.json"
    target.write_text(json.dumps(result)+"\n")

class _Instrument(ast.NodeTransformer):
    def __init__(self):
        self.calls = 0
    def visit_Assign(self,node):
        if (isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                and node.value.func.attr == "draw" and isinstance(node.value.func.value,ast.Name)
                and node.value.func.value.id == "markov_chain"):
            self.calls += 1
            before = ast.parse("_bench_tm_w0=time.perf_counter_ns()\n_bench_tm_c0=time.thread_time_ns()").body
            after = ast.parse("_bench_tm_c1=time.thread_time_ns()\n_bench_tm_w1=time.perf_counter_ns()\n_bench_tm_draw_wall+=_bench_tm_w1-_bench_tm_w0\n_bench_tm_draw_cpu+=_bench_tm_c1-_bench_tm_c0\n_bench_tm_calls+=1").body
            return before+[node]+after
        return self.generic_visit(node)
    def visit_Return(self,node):
        record = ast.parse("_emit(chain_idx,n_samples,thinning,_bench_tm_draw_wall,_bench_tm_draw_cpu,_bench_tm_calls,_bench_tm_entry)").body
        return record+[node]

def _make(name,new_name):
    source = inspect.getsource(getattr(_misc,name))
    tree = ast.parse(source)
    original = ast.dump(tree,include_attributes=False)
    assert "_bench_tm_" not in source
    transform = _Instrument()
    tree = transform.visit(tree)
    assert transform.calls == 1
    tree.body[0].name = new_name
    setup = ast.parse("_bench_tm_entry=time.perf_counter_ns()\n_bench_tm_draw_wall=0\n_bench_tm_draw_cpu=0\n_bench_tm_calls=0").body
    tree.body[0].body = setup+tree.body[0].body
    ast.fix_missing_locations(tree)
    exec(compile(tree,__file__,"exec"),globals())
    PROOF[name] = dict(original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      original_ast_sha256=hashlib.sha256(original.encode()).hexdigest(),
                      instrumented_ast_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
                      instrumented_draw_call_sites=transform.calls,
                      transformation="Original worker body preserved; timer statements around its one draw call site and one log emission before return.")

_make("_sample_parallel_chain","_timed_parallel_chain")
_make("_sequential_sampling","_timed_sequential_sampling")

def install(directory):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    os.environ["BENCH_HOPSY_WORKER_TIMING_DIR"] = str(directory.resolve())
    _misc._sample_parallel_chain = _timed_parallel_chain
    _misc._sequential_sampling = _timed_sequential_sampling
    return dict(hopsy_misc_sha256=hashlib.sha256(Path(_misc.__file__).read_bytes()).hexdigest(),
                instrumentation_module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),functions=PROOF)

def summarize(directory,expected_chains,expected_draws):
    records = [json.loads(path.read_text()) for path in Path(directory).glob("pid*-chain*.json")]
    assert len({row["chain_idx"] for row in records}) == expected_chains
    assert sum(row["draw_calls"] for row in records) == expected_chains*expected_draws
    workers = {}
    for row in records:
        worker = workers.setdefault(row["pid"],dict(draw_call_wall_seconds=0.,draw_call_thread_cpu_seconds=0.,chain_tasks=0))
        worker["draw_call_wall_seconds"] += row["draw_call_wall_seconds"]
        worker["draw_call_thread_cpu_seconds"] += row["draw_call_thread_cpu_seconds"]
        worker["chain_tasks"] += 1
    return dict(status="complete",chain_tasks=len(records),worker_count=len(workers),
                worker_draw_critical_path_seconds=max(row["draw_call_wall_seconds"] for row in workers.values()),
                worker_draw_cpu_critical_path_seconds=max(row["draw_call_thread_cpu_seconds"] for row in workers.values()),
                total_worker_draw_thread_cpu_seconds=sum(row["draw_call_thread_cpu_seconds"] for row in workers.values()),
                workers=workers,
                critical_path_definition="Maximum over worker PIDs of accumulated MarkovChain.draw call durations. Gaps between tasks/calls, serialization and IPC are excluded; elapsed intervals include OS descheduling and Python binding/state-copy overhead. This is a compute critical-path proxy, not pure arithmetic-only time.")
