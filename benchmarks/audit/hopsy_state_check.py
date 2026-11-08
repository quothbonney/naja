"""Check that repeated hopsy.sample calls preserve chains and RNG state."""
import json
from pathlib import Path
import hopsy
import numpy as np


def generate(procs, chunks):
    problem=hopsy.Problem(np.vstack([np.eye(3),-np.eye(3)]),np.ones(6))
    chains=[hopsy.MarkovChain(problem,proposal=hopsy.UniformCoordinateHitAndRunProposal,
                             starting_point=np.zeros(3)) for _ in range(2)]
    rngs=[hopsy.RandomNumberGenerator(101+i) for i in range(2)]
    blocks=[]
    for n in chunks:
        _,x=hopsy.sample(chains,rngs,n_samples=n,thinning=2,n_procs=procs,progress_bar=False)
        blocks.append(x)
    return np.concatenate(blocks,axis=1)


def main():
    full=generate(1,[64]); chunked=generate(1,[32,32]); parallel=generate(2,[64])
    result={'hopsy_version':hopsy.__version__,'shape':list(full.shape),
            'full_vs_chunked_bitwise_equal':bool(np.array_equal(full,chunked)),
            'serial_vs_parallel_single_call_bitwise_equal':bool(np.array_equal(full,parallel)),
            'different_chain_outputs':bool(not np.array_equal(full[0],full[1])),
            'parallel_multichunk_caveat':'Second parallel call failed hopsy model unpickling in a separate audit; current benchmark wrappers use one full chunk.'}
    assert all(result[k] for k in ['full_vs_chunked_bitwise_equal','serial_vs_parallel_single_call_bitwise_equal','different_chain_outputs'])
    (Path(__file__).parent/'hopsy-state-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
