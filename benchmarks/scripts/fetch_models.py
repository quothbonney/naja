#!/usr/bin/env python3
"""Freeze the proposed nine-model panel from the official legacy BiGG archive."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import urllib.request

MODELS = ('e_coli_core', 'iAB_RBC_283', 'iJR904', 'iAF1260', 'iJO1366',
          'iML1515', 'iMM904', 'RECON1', 'Recon3D')
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', nargs='*', default=list(MODELS))
    parser.add_argument('--out', type=Path, default=ROOT / 'data' / 'raw')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / 'manifest.json'
    records = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for model in args.models:
        if model not in MODELS:
            parser.error(f'unknown panel model: {model}')
        url = f'https://biggdata.ucsd.edu/static/models/{model}.json'
        path = args.out / f'{model}.json'
        if path.exists() and model in records:
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != records[model]['sha256']:
                raise RuntimeError(f'checksum mismatch: {path}')
            print(f'{model}: cached, checksum verified', flush=True)
            continue
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read()
        obj = json.loads(data)
        if not obj.get('reactions') or not obj.get('metabolites'):
            raise ValueError(f'invalid COBRA model from {url}')
        temporary = path.with_suffix('.json.partial')
        temporary.write_bytes(data)
        temporary.replace(path)
        records[model] = dict(url=url, retrieved_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                              sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                              reactions=len(obj['reactions']), metabolites=len(obj['metabolites']),
                              genes=len(obj.get('genes', [])), model_id=obj.get('id'))
        manifest_path.write_text(json.dumps(records, indent=2, sort_keys=True) + '\n')
        print(f'{model}: downloaded {len(data)} bytes', flush=True)

if __name__ == '__main__':
    main()
