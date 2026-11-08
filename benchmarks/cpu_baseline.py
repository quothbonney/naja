#!/usr/bin/env python3
"""PolyRound preprocessing and hopsy CHRR benchmark, sharing Naja bundles."""
from __future__ import annotations
import time
PROCESS_STARTED = time.perf_counter()
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np


def versions():
    result = {p: importlib.metadata.version(p) for p in
              ("hopsy", "PolyRound", "cobra", "numpy", "scipy", "swiglpk")}
    for p in ("highspy", "osqp"):
        try:
            result[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str) + "\n")


def preprocess(args):
    import cobra
    from PolyRound.api import PolyRoundApi
    from PolyRound.settings import PolyRoundSettings
    if args.fast_sparse_assembly:
        from fast_sparse import enable
        enable()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "rounding/polytope.npz").exists() and not args.force:
        raise RuntimeError(f"Refusing to redo cached preprocessing: {out}; use --force")
    timings = {}
    t0 = time.perf_counter()
    cobra.Configuration().solver = "glpk"
    model = cobra.io.load_json_model(str(args.model))
    model.solver = "glpk"
    timings["load_model_seconds"] = time.perf_counter() - t0
    full_model=model
    ids=[r.id for r in full_model.reactions]
    n = len(model.reactions)
    lower = np.array([r.lower_bound for r in model.reactions])
    upper = np.array([r.upper_bound for r in model.reactions])
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError("Infinite model bounds: refusing implicit clipping")
    pruning=None
    pruned=[]
    if args.zero_prune_fva:
        import pandas as pd
        cache=Path(args.zero_prune_fva)
        pruning=json.loads((cache/'fva.json').read_text())
        if pruning['source_sha256']!=hashlib.sha256(Path(args.model).read_bytes()).hexdigest():
            raise ValueError('FVA source SHA does not match model')
        if not pruning['objective_zero'] or pruning['open_exchanges'] or pruning['fraction_of_optimum']!=0 or pruning['bounds_modified']:
            raise ValueError('FVA cache changed target objective or exchanges')
        ranges=pd.read_csv(cache/'ranges.csv',index_col=0).loc[ids]
        finite=np.isfinite(ranges[['minimum','maximum']]).all(axis=1)
        pruned=ranges.index[finite & (ranges[['minimum','maximum']].abs().max(axis=1)<=1e-9)].tolist()
        model=full_model.copy()
        model.remove_reactions(pruned,remove_orphans=False)
        pruning={**pruning,'cache':str(cache.resolve()),'zero_cutoff':1e-9,
                 'zero_pruned_reactions':pruned,'active_reactions':len(model.reactions)}
        timings['zero_prune_fva_seconds']=pruning['wall_seconds']
        print(f'Zero-pruning: {len(pruned)} / {n} reactions at 1e-9; bounds unchanged',flush=True)
    t = time.perf_counter()
    poly = PolyRoundApi.cobra_model_to_polytope(model)
    timings["construct_polytope_seconds"] = time.perf_counter() - t
    settings = PolyRoundSettings(backend=args.backend, verbose=args.verbose,
                                presolve=args.presolve,
                                reduce=not args.no_redundancy_reduction,
                                hp_flags={"FeasibilityTol": args.feasibility_tol})
    face_pruning=None
    if args.fva_inactive_faces:
        if not pruning: raise ValueError('FVA face certificates require --zero-prune-fva')
        active=[r.id for r in model.reactions];k=len(active)
        fva=ranges.loc[active]
        widths=fva.maximum-fva.minimum
        if not np.isfinite(fva.to_numpy()).all() or widths.min()<=settings.thresh:
            raise ValueError('FVA does not certify positive width in every active reaction')
        t=time.perf_counter()
        lo=np.array([r.lower_bound for r in model.reactions]);hi=np.array([r.upper_bound for r in model.reactions])
        slack=np.r_[hi-fva.maximum.to_numpy(),fva.minimum.to_numpy()-lo]
        keep=slack<=1e-3
        values=poly.A.values
        if values.shape!=(2*k,k) or np.count_nonzero(values)!=2*k or not np.all(values[np.arange(k),np.arange(k)]==1) or not np.all(values[k+np.arange(k),np.arange(k)]==-1):
            raise ValueError('Unexpected PolyRound bounds-row ordering')
        poly.A=poly.A.iloc[keep].copy();poly.b=poly.b.iloc[keep].copy();poly.normalize()
        face_pruning=dict(margin=1e-3,retained=int(keep.sum()),removed=int((~keep).sum()),
                          min_coordinate_width=float(widths.min()),
                          min_removed_slack=float(slack[~keep].min()) if (~keep).any() else None,
                          simplify_skipped=True,certificate='Full original-target FVA; strict inactivity and positive coordinate widths')
        timings['fva_face_filter_seconds']=time.perf_counter()-t
        print('FVA face certificates: '+json.dumps(face_pruning),flush=True)
    prep_start = time.perf_counter()
    t = time.perf_counter()
    if not face_pruning: poly = PolyRoundApi.simplify_polytope(poly, settings)
    timings["simplify_seconds"] = time.perf_counter() - t
    t = time.perf_counter()
    if not poly.inequality_only:
        poly = PolyRoundApi.transform_polytope(poly, settings)
    timings["equality_elimination_seconds"] = time.perf_counter() - t
    t = time.perf_counter()
    poly = PolyRoundApi.round_polytope(poly, settings)
    timings["round_seconds"] = time.perf_counter() - t
    timings["polyround_seconds"] = time.perf_counter() - prep_start
    A = np.ascontiguousarray(poly.A.values, dtype=np.float64)
    b = np.ascontiguousarray(poly.b.values, dtype=np.float64)
    T = np.ascontiguousarray(poly.transformation.values, dtype=np.float64)
    shift = np.ascontiguousarray(poly.shift.values, dtype=np.float64)
    start = np.zeros(A.shape[1], dtype=np.float64)
    if not np.all(b - A @ start > 0):
        raise ValueError("PolyRound did not produce strictly interior origin")
    # Ensure transformation row order matches the original model's reaction IDs.
    mapping_ids = list(poly.transformation.index)
    if mapping_ids != ids:
        if set(mapping_ids)!=set(ids)-set(pruned):
            raise ValueError('Unexpected reaction mapping after preprocessing')
        full_T=np.zeros((n,A.shape[1]),dtype=np.float64)
        full_shift=np.zeros(n,dtype=np.float64)
        positions={r:i for i,r in enumerate(ids)}
        order=[positions[r] for r in mapping_ids]
        full_T[order]=T;full_shift[order]=shift
        T,shift=full_T,full_shift
    rounding, gem = out / "rounding", out / "gem"
    rounding.mkdir(exist_ok=True)
    gem.mkdir(exist_ok=True)
    t = time.perf_counter()
    np.savez(rounding / "polytope.npz", A=A, b=b, start=start, T=T, shift=shift)
    np.savetxt(gem / "l_bounds.csv", lower, delimiter=",")
    np.savetxt(gem / "u_bounds.csv", upper, delimiter=",")
    np.savetxt(gem / "A_eq.csv", cobra.util.array.create_stoichiometric_matrix(full_model), delimiter=",")
    np.savetxt(gem / "b_eq.csv", np.zeros(len(full_model.metabolites)), delimiter=",")
    (gem / "reaction_ids.txt").write_text("\n".join(ids) + "\n")
    timings["write_seconds"] = time.perf_counter() - t
    timings["total_seconds"] = time.perf_counter() - t0
    timings['total_preprocessing_seconds']=timings['total_seconds']+timings.get('zero_prune_fva_seconds',0.)
    meta = dict(model_id=model.id, n=n, d=A.shape[1], m=A.shape[0],
                original_metabolites=len(full_model.metabolites),
                source=str(Path(args.model).resolve()),
                source_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                solver=args.backend, versions=versions(), timings=timings,
                settings=vars(settings), objective_constraint="none; original model bounds",
                preprocessing_variant='fva_inactive_faces' if face_pruning else ('zero_fva_pruned' if pruning else 'standard'),
                zero_pruning=pruning,fva_face_pruning=face_pruning,
                fast_sparse_assembly=args.fast_sparse_assembly,
                fast_sparse_assembly_sha256=hashlib.sha256((Path(__file__).parent/'fast_sparse.py').read_bytes()).hexdigest() if args.fast_sparse_assembly else None,
                min_start_slack=float(b.min()), command=sys.argv,
                hostname=platform.node())
    save_json(out / "meta.json", meta)
    save_json(rounding / "manifest.json", dict(format_version=2, model_name=out.name,
                                               rounding_dim=A.shape[1], n_reactions=n,
                                               n_constraints=A.shape[0]))
    print(json.dumps(meta, indent=2, default=str), flush=True)


def thinning_value(label, dim):
    if label == "default":
        return 1, "hopsy.sample API default thinning=1 (unthinned)"
    if label == "jadebeck":
        return max(1, int(math.ceil(dim * dim / 6))), "ceil(d^2/6), Jadebeck et al. 2023 Eq.14; GEM guideline"
    if label == "legacy":
        return max(1, 8 * dim * dim), "8*d^2, legacy CHRR guideline; NOT hopsy API default"
    return max(1, int(label)), "explicit fixed thinning"


def sample(args):
    import hopsy
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "samples.npy").exists() and not args.force:
        raise RuntimeError(f"Output already exists: {out}; use --force")
    timings = dict(load_seconds=0., initialize_seconds=0., sample_seconds=0.,
                   write_seconds=0.)
    wall_start = time.perf_counter()
    t = time.perf_counter()
    data = np.load(Path(args.model) / "rounding/polytope.npz")
    A, b, start = data["A"], data["b"], data["start"]
    timings["load_seconds"] = time.perf_counter() - t
    thin, formula = thinning_value(args.thinning, A.shape[1])
    total = args.draws + args.warmup
    chunk = args.chunk or total
    if min(args.chains,args.workers)>1 and chunk<total:
        raise ValueError('hopsy 1.7 parallel repeated calls can fail unpickling; use --chunk >= draws+warmup (default) or --workers 1')
    t = time.perf_counter()
    problem = hopsy.Problem(A, b)
    chains = [hopsy.MarkovChain(problem, proposal=hopsy.UniformCoordinateHitAndRunProposal,
                              starting_point=start.copy()) for _ in range(args.chains)]
    rngs = [hopsy.RandomNumberGenerator(args.seed + i) for i in range(args.chains)]
    timings["initialize_seconds"] = time.perf_counter() - t
    output = np.lib.format.open_memmap(out / "samples.npy", mode="w+", dtype=np.float64,
                                     shape=(args.chains, args.draws, A.shape[1]))
    package_versions=versions()
    module_sha=hashlib.sha256(Path(sys.modules['hopsy.core'].__file__).read_bytes()).hexdigest()
    signature_data=dict(versions=package_versions,python=platform.python_version(),
                        build_variant=args.build_variant,module_sha256=module_sha,
                        flags=os.environ.get('HOPSY_CXXFLAGS','wheel defaults'))
    status = dict(method="hopsy_chrr", model=str(Path(args.model).resolve()),
                  build_variant=args.build_variant, hopsy_build=hopsy.__build__,
                  hopsy_cxxflags=os.environ.get('HOPSY_CXXFLAGS','wheel defaults'),
                  hopsy_module_sha256=module_sha,
                  environment_signature=hashlib.sha256(json.dumps(signature_data,sort_keys=True).encode()).hexdigest(),
                  build_manifest=os.environ.get('HOPSY_BUILD_MANIFEST'),
                  seed=args.seed, chain_seeds=[args.seed+i for i in range(args.chains)],
                  chains=args.chains, draws=args.draws, warmup=args.warmup,
                  thinning=thin, thinning_label=args.thinning, thinning_formula=formula,
                  sample_space="reduced", shape=list(output.shape), dtype="float64",
                  workers=min(args.chains, args.workers), requested_workers=args.workers,
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                  allocated_cpus=os.environ.get('SLURM_CPUS_PER_TASK'),
                  cpu_affinity=sorted(os.sched_getaffinity(0)),
                  hostname=platform.node(), versions=package_versions, command=sys.argv,
                  status="running", completed_draws=0, timings=timings,
                  includes_backmap=False, initial_state="shared rounded origin; independent RNGs")
    save_json(out / "run.json", status)
    completed = retained = 0
    acceptance = []
    while completed < total:
        count = min(chunk, total-completed)
        t = time.perf_counter()
        rates, samples = hopsy.sample(chains, rngs, n_samples=count, thinning=thin,
                                     n_procs=min(args.chains, args.workers), progress_bar=False)
        timings["sample_seconds"] += time.perf_counter() - t
        acceptance.append(np.asarray(rates).tolist())
        # Discard exactly the requested retained warmup draws, across chunk boundaries.
        skip = max(0, min(count, args.warmup-completed))
        keep = samples[:, skip:, :]
        t = time.perf_counter()
        if keep.shape[1]:
            output[:, retained:retained+keep.shape[1], :] = keep
            retained += keep.shape[1]
            output.flush()
        timings["write_seconds"] += time.perf_counter() - t
        completed += count
        status["completed_draws"] = retained
        status["completed_total_draws"] = completed
        timings["wall_seconds"] = time.perf_counter() - wall_start
        save_json(out / "run.json", status)
        print(json.dumps(dict(progress=completed, total=total, sample_seconds=timings["sample_seconds"])), flush=True)
    status["acceptance_rates_by_chunk"] = acceptance
    status["status"] = "complete"
    timings["wall_seconds"] = time.perf_counter() - wall_start
    status["process_wall_seconds"] = time.perf_counter() - PROCESS_STARTED
    status["retained_samples_per_second"] = args.chains * args.draws / timings["wall_seconds"]
    save_json(out / "run.json", status)
    print(json.dumps(status, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prep")
    prep.add_argument("--model", required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--force", action="store_true")
    prep.add_argument("--verbose", action="store_true")
    prep.add_argument("--presolve", action="store_true")
    prep.add_argument("--backend", choices=("glpk", "hybrid"), default="glpk")
    prep.add_argument("--no-redundancy-reduction", action="store_true")
    prep.add_argument("--feasibility-tol", type=float, default=1e-9)
    prep.add_argument("--zero-prune-fva",help='Cache directory from zero_prune_fva.py; preserves full reaction backmap')
    prep.add_argument("--fva-inactive-faces",action='store_true',help='Use cached full-FVA face certificates instead of serial simplification')
    prep.add_argument("--fast-sparse-assembly",action='store_true',help='Process-local bulk CSR row assembly; coefficient values unchanged')
    prep.set_defaults(func=preprocess)
    draw = subs.add_parser("sample")
    draw.add_argument("--model", required=True)
    draw.add_argument("--out", required=True)
    draw.add_argument("--seed", type=int, default=1)
    draw.add_argument("--chains", type=int, default=16)
    draw.add_argument("--draws", type=int, default=10000)
    draw.add_argument("--warmup", type=int, default=1000)
    draw.add_argument("--thinning", default="jadebeck")
    draw.add_argument("--workers", type=int, default=16)
    draw.add_argument("--build-variant", choices=("wheel", "native", "source_generic"), default="wheel")
    draw.add_argument("--chunk", type=int, default=None)
    draw.add_argument("--force", action="store_true")
    draw.set_defaults(func=sample)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
