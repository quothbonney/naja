"""Exit2 on cache miss; never synthesize an unknown producer identity."""
import argparse
from pathlib import Path
import sys

B = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B))
import metrics
from quality_diagnostics import reuse,save_identity

def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--wall',type=float,required=True)
    a=p.parse_args()
    if not (a.reference/'diagnostics_identity.json').exists():return 2
    if not reuse(a.samples,a.model,a.out,a.wall,a.reference):return 2
    save_identity(a.samples,a.model,a.out)
    return 0

if __name__=='__main__':raise SystemExit(main())
