#!/usr/bin/env python3
"""Create naja rounding *bundles* (polytope.npz / extra.npz + manifest.json).

The bundle format replaces the legacy per-file CSV rounding layout:

    rounding/<model>_rounding_A.csv, _b.csv, _start.csv, _T.csv, _shift.csv
    rounding/<model>_rounding_extra_A.csv, _extra_b.csv

with:

    rounding/polytope.npz   arrays: A, b, start, T, shift  (shared base rounding)
    rounding/extra.npz      arrays: extra_A, extra_b       (per-model KO delta, optional)
    rounding/manifest.json  metadata

naja reads either layout (it prefers a bundle when polytope.npz is present), so
conversion can be done lazily, model by model.

Two entry points:

  * Command line, to convert existing model directories:

        # one model
        python tools/pack_model.py convert --model-dir /path/to/models/wt

        # many, from a model list (one name per line)
        python tools/pack_model.py convert-all \
            --models-root /path/to/models --model-list jobs.txt

    Add --remove-legacy to delete the CSVs after a verified round-trip, and
    --compress to write deflate-compressed npz (smaller; naja reads both).

  * Imported from the rounding pipeline, to emit a bundle directly from arrays,
    skipping CSV entirely:

        from pack_model import write_bundle
        write_bundle(rounding_dir, "wt", A=A, b=b, start=start, T=T, shift=shift)

Arrays are always stored float64, C-contiguous. Vectors are 1-D (shape (n,)).
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np

FORMAT_VERSION = 2

# Logical array name -> legacy CSV suffix (after "<model>_rounding").
_POLYTOPE_ARRAYS = {
    "A": "_A.csv",
    "b": "_b.csv",
    "start": "_start.csv",
    "T": "_T.csv",
    "shift": "_shift.csv",
}
_EXTRA_ARRAYS = {
    "extra_A": "_extra_A.csv",
    "extra_b": "_extra_b.csv",
}


def _as_matrix(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 0:
        a = a.reshape(1, 1)
    elif a.ndim == 1:
        a = a.reshape(-1, 1)
    return np.ascontiguousarray(a)


def _as_vector(a):
    a = np.asarray(a, dtype=np.float64).ravel()
    return np.ascontiguousarray(a)


def _load_csv_matrix(path):
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    return np.ascontiguousarray(a, dtype=np.float64)


def _load_csv_vector(path):
    a = np.loadtxt(path, delimiter=",")
    return _as_vector(a)


def write_bundle(
    rounding_dir,
    model_name,
    A,
    b,
    start,
    T=None,
    shift=None,
    extra_A=None,
    extra_b=None,
    compress=False,
    source="native",
    manifest_extra=None,
):
    """Write polytope.npz (+ extra.npz + manifest.json) from in-memory arrays.

    T/shift are optional but must be provided together (they are required for
    backmapping to reaction space). extra_A/extra_b likewise pair together.
    Returns the manifest dict that was written.
    """
    os.makedirs(rounding_dir, exist_ok=True)

    A = _as_matrix(A)
    b = _as_vector(b)
    start = _as_vector(start)
    if b.shape[0] != A.shape[0]:
        raise ValueError(f"b length {b.shape[0]} != A rows {A.shape[0]}")
    if start.shape[0] != A.shape[1]:
        raise ValueError(f"start length {start.shape[0]} != A cols {A.shape[1]}")

    poly = {"A": A, "b": b, "start": start}

    has_backmap = T is not None or shift is not None
    if has_backmap:
        if T is None or shift is None:
            raise ValueError("T and shift must be provided together")
        T = _as_matrix(T)
        shift = _as_vector(shift)
        if T.shape[1] != A.shape[1]:
            raise ValueError(f"T cols {T.shape[1]} != A cols {A.shape[1]}")
        if shift.shape[0] != T.shape[0]:
            raise ValueError(f"shift length {shift.shape[0]} != T rows {T.shape[0]}")
        poly["T"] = T
        poly["shift"] = shift

    saver = np.savez_compressed if compress else np.savez
    saver(os.path.join(rounding_dir, "polytope.npz"), **poly)

    has_extra = extra_A is not None or extra_b is not None
    n_extra = 0
    extra_path = os.path.join(rounding_dir, "extra.npz")
    if has_extra:
        if extra_A is None or extra_b is None:
            raise ValueError("extra_A and extra_b must be provided together")
        extra_A = _as_matrix(extra_A)
        extra_b = _as_vector(extra_b)
        if extra_A.shape[0] != extra_b.shape[0]:
            raise ValueError("extra_A rows != extra_b length")
        if extra_A.shape[1] != A.shape[1]:
            raise ValueError("extra_A cols != A cols")
        n_extra = int(extra_A.shape[0])
        saver(extra_path, extra_A=extra_A, extra_b=extra_b)
    elif os.path.exists(extra_path):
        # No delta for this model: make sure a stale extra.npz can't linger.
        os.remove(extra_path)

    manifest = {
        "format_version": FORMAT_VERSION,
        "model_name": model_name,
        "rounding_dim": int(A.shape[1]),
        "n_constraints": int(A.shape[0]),
        "n_reactions": int(T.shape[0]) if has_backmap else None,
        "n_extra_constraints": n_extra,
        "has_backmap": bool(has_backmap),
        "compressed": bool(compress),
        "source": source,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    with open(os.path.join(rounding_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def _legacy_prefix(rounding_dir, model_name):
    return os.path.join(rounding_dir, f"{model_name}_rounding")


def convert_model(model_dir, remove_legacy=False, compress=False):
    """Convert one legacy model directory in place to a bundle."""
    model_dir = os.path.abspath(model_dir.rstrip("/"))
    model_name = os.path.basename(model_dir)
    rounding_dir = os.path.join(model_dir, "rounding")
    if not os.path.isdir(rounding_dir):
        raise FileNotFoundError(f"no rounding/ dir under {model_dir}")

    prefix = _legacy_prefix(rounding_dir, model_name)

    def req(suffix):
        p = prefix + suffix
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing legacy file: {p}")
        return p

    A = _load_csv_matrix(req("_A.csv"))
    b = _load_csv_vector(req("_b.csv"))
    start = _load_csv_vector(req("_start.csv"))

    T = shift = None
    if os.path.exists(prefix + "_T.csv"):
        T = _load_csv_matrix(prefix + "_T.csv")
        shift = _load_csv_vector(req("_shift.csv"))

    extra_A = extra_b = None
    if os.path.exists(prefix + "_extra_A.csv") and os.path.exists(prefix + "_extra_b.csv"):
        extra_A = _load_csv_matrix(prefix + "_extra_A.csv")
        extra_b = _load_csv_vector(prefix + "_extra_b.csv")

    manifest = write_bundle(
        rounding_dir, model_name, A=A, b=b, start=start,
        T=T, shift=shift, extra_A=extra_A, extra_b=extra_b,
        compress=compress, source="converted_from_csv",
    )

    # Round-trip verification before we consider deleting anything.
    _verify_roundtrip(rounding_dir, A, b, start, T, shift, extra_A, extra_b)

    if remove_legacy:
        suffixes = list(_POLYTOPE_ARRAYS.values()) + list(_EXTRA_ARRAYS.values())
        for suf in suffixes:
            p = prefix + suf
            # Legacy files may be symlinks (inherited rounding); unlink either way.
            if os.path.islink(p) or os.path.exists(p):
                os.remove(p)

    return manifest


def _verify_roundtrip(rounding_dir, A, b, start, T, shift, extra_A, extra_b):
    z = np.load(os.path.join(rounding_dir, "polytope.npz"))
    assert np.allclose(z["A"], A), "A round-trip mismatch"
    assert np.allclose(z["b"], b), "b round-trip mismatch"
    assert np.allclose(z["start"], start), "start round-trip mismatch"
    if T is not None:
        assert np.allclose(z["T"], T), "T round-trip mismatch"
        assert np.allclose(z["shift"], shift), "shift round-trip mismatch"
    if extra_A is not None:
        ze = np.load(os.path.join(rounding_dir, "extra.npz"))
        assert np.allclose(ze["extra_A"], extra_A), "extra_A round-trip mismatch"
        assert np.allclose(ze["extra_b"], extra_b), "extra_b round-trip mismatch"


def _read_model_list(path):
    names = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    if not names:
        raise ValueError(f"no model names in {path}")
    return names


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c1 = sub.add_parser("convert", help="convert one model directory")
    c1.add_argument("--model-dir", required=True)
    c1.add_argument("--remove-legacy", action="store_true",
                    help="delete legacy CSVs after a verified round-trip")
    c1.add_argument("--compress", action="store_true",
                    help="write deflate-compressed npz (naja reads both)")

    c2 = sub.add_parser("convert-all", help="convert many models from a list")
    c2.add_argument("--models-root", required=True)
    c2.add_argument("--model-list", required=True)
    c2.add_argument("--remove-legacy", action="store_true")
    c2.add_argument("--compress", action="store_true")

    args = ap.parse_args(argv)

    if args.cmd == "convert":
        m = convert_model(args.model_dir, args.remove_legacy, args.compress)
        print(f"OK  {m['model_name']}  d={m['rounding_dim']} "
              f"constraints={m['n_constraints']} extra={m['n_extra_constraints']} "
              f"backmap={m['has_backmap']}")
        return 0

    if args.cmd == "convert-all":
        names = _read_model_list(args.model_list)
        ok = 0
        for name in names:
            model_dir = os.path.join(args.models_root, name)
            try:
                convert_model(model_dir, args.remove_legacy, args.compress)
                ok += 1
            except Exception as e:  # noqa: BLE001 - report and continue
                print(f"FAIL {name}: {e}", file=sys.stderr)
        print(f"converted {ok} / {len(names)}")
        return 0 if ok == len(names) else 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
