# Coordinate-update accounting

One hopsy thinning unit is **one coordinate update**, not one full coordinate sweep. Python `sample` calls `markov_chain.draw(rng, thinning)` once per retained draw; HOPS `MarkovChainAdapter` executes `thinning` calls to the draw implementation; the uniform draw adapter proposes and accepts once; the coordinate proposal advances exactly one coordinate.

Hopsy cycles deterministically through coordinates (`++coordinateToUpdate %= state.rows()`); Naja randomly selects a coordinate. They target the same uniform measure but have different scan kernels. The native hopsy module was also checked on a3-Dbox: successive thinning-1 calls changed coordinates1,2,0,1,2,0.

Task3 performs16 × (10,000+1,000) ×96 = **16,896,000 actual coordinate updates**, including warmup. Retained-only updates are15,360,000. Median pooled reaction ESS /16.896 is ESS per million actual updates.

Full source snapshots, official-sdist SHA, native-module SHA and the smoke-test observations are in [thinning_semantics.json](thinning_semantics.json). No estimator or sampling code was modified.
