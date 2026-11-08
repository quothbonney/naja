"""Exploratory equality-only bundle; deliberately skips PolyRound MVE rounding.

This never replaces the canonical Recon3D bundle or contributes a rounded-model
CPU/GPU speedup. Original bounds and the existing FVA inactivity certificates
are retained. GPU barrier preprocessing is measured in a separate later run.
"""
from argparse import Namespace
import json
from pathlib import Path
import sys

B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B))
import cobra  # Initialize COBRA before PolyRound's optional-dependency import.
from PolyRound.api import PolyRoundApi
from cpu_baseline import preprocess

out = B / 'data/models/Recon3D_nullspace_only'
PolyRoundApi.round_polytope = staticmethod(lambda polytope, settings: polytope)
preprocess(Namespace(model=str(B / 'data/raw/Recon3D.json'), out=str(out),
    force=False, verbose=True, presolve=False, backend='glpk',
    no_redundancy_reduction=False, feasibility_tol=1e-9,
    zero_prune_fva=str(B / 'data/preprocessing/Recon3D_zero_fva'),
    fva_inactive_faces=True, fast_sparse_assembly=True))
meta_path = out / 'meta.json'
meta = json.loads(meta_path.read_text())
meta.update(preprocessing_variant='exploratory_equality_only_no_mve',
    polyround_mve_performed=False,
    benchmark_scope='Exploratory barrier-only feasibility and throughput; excluded from canonical rounded-model comparisons.')
meta['timings']['round_seconds'] = 0.0
meta_path.write_text(json.dumps(meta, indent=2) + '\n')
(out / 'EXPLORATORY_NO_MVE.txt').write_text(meta['benchmark_scope'] + '\n')
