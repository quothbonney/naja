#!/usr/bin/env python3
"""Copy publication evidence into the source checkout without raw sample payloads.

Run from the original benchmark workspace. Does not stage, commit, or push.
"""
import csv
import hashlib
import re
import shutil
from pathlib import Path

W = Path(__file__).resolve().parents[2]
DEST = W / 'naja-test-jack/naja'
EXTENSIONS = {
    '.md', '.csv', '.json', '.jsonl', '.txt', '.py', '.sh', '.sbatch',
    '.svg', '.pdf', '.png', '.patch', '.log', '.sha256', '.psv',
    '.cpp', '.cu', '.cuh', '.h', '.gz',
}
MAX_BYTES = 1_000_000
SECRET = re.compile(rb'(?i)(?:WLSSECRET|WLSACCESSID|GRB_WLSSECRET|GRB_WLSACCESSID)[\"\x27]?\s*[=:]\s*[\"\x27]?([A-Za-z0-9-]{20,})')


def main():
    candidates = [W / 'CLOSEOUT.md']
    candidates += [p for p in (W / 'benchmarks').iterdir() if p.is_file()]
    for folder in ('scripts', 'results', 'audit', 'bin'):
        candidates += list((W / 'benchmarks' / folder).rglob('*'))
    included, omitted, seen = [], [], set()
    for p in sorted(candidates):
        if not p.is_file() or p in seen:
            continue
        seen.add(p)
        rel = p.relative_to(W)
        size = p.stat().st_size
        reason = None
        if p.is_symlink():
            reason = 'symlink; original target remains in workspace'
        elif '__pycache__' in p.parts or p.suffix not in EXTENSIONS:
            reason = 'raw data, profiler binary, cache, executable, or runtime state'
        elif size > MAX_BYTES:
            reason = 'payload exceeds 1 MB Git evidence limit'
        elif any(x in p.name.lower() for x in ('gurobi.lic', 'credential', 'private_key')):
            reason = 'credential filename'
        if reason:
            omitted.append((str(rel), size, reason))
            continue
        payload = p.read_bytes()
        if SECRET.search(payload) or re.search(rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}', payload):
            raise SystemExit(f'Credential assignment detected; refusing package: {rel}')
        target = DEST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or (target.stat().st_size, target.stat().st_mtime_ns) != (size, p.stat().st_mtime_ns):
            shutil.copy2(p, target)
        included.append((str(rel), size, hashlib.sha256(payload).hexdigest()))
    out = DEST / 'benchmarks/publication-package'
    out.mkdir(parents=True, exist_ok=True)
    for name, header, rows in (
        ('included.csv', ('path', 'bytes', 'sha256'), included),
        ('server-only.csv', ('path', 'bytes', 'reason'), omitted),
    ):
        with (out / name).open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
    (out / 'README.md').write_text(
        '# Preprint evidence package\n\n'
        'Start with ../../CLOSEOUT.md (repository root) and '
        '../results/RESULT_FILES.txt. Reports preserve their original paths.\n\n'
        'The package includes benchmark drivers, dependency lockfiles, reports, '
        'tables, figures, logs and provenance files up to 1 MB each. '
        'included.csv records SHA256 hashes of copied source files. '
        'server-only.csv inventories omitted original payloads and their sizes. '
        'Raw samples, prepared model arrays, profiler databases, environments, '
        'binaries and credentials are not deposited in Git. Original payloads '
        f'remain under `{W}`. Existing absolute paths identify that server snapshot; '
        'commands can require path adjustment on another machine.\n\n'
        'Historical partial, failed and exploratory runs are retained as evidence; '
        'their inclusion does not promote them into final paper results. '
        'Historical controller scripts are archived for reproducibility and '
        'should not be restarted automatically. No new jobs are launched by packaging.\n'
    )
    print(f'Packaged {len(included)} files, {sum(x[1] for x in included):,} bytes; '
          f'{len(omitted)} server-only files inventoried.')


if __name__ == '__main__':
    main()
