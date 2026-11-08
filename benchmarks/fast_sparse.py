"""Exact-value bulk row assembly for PolyRound's optlang LP builder.

The upstream implementation indexes CSR once per coefficient. Iterating each
canonical CSR row's stored values avoids that Python/SciPy scalar-index cost.
"""
from PolyRound.static_classes import lp_interfacing as lp


def bulk_constraint_system(model,A,b,x,backend,equality=False):
    interface=lp.solvers[backend]
    matrix=lp.sp.csr_matrix(A,copy=True)
    matrix.sum_duplicates();matrix.eliminate_zeros();matrix.sort_indices()
    rows={}
    for i,bound in enumerate(b):
        name=b.index[i] if isinstance(b,lp.pd.Series) else lp.uuid.uuid4().hex
        if not isinstance(name,str): name=lp.uuid.uuid4().hex
        begin,end=matrix.indptr[i:i+2]
        row=interface.Constraint(lp.Zero,ub=bound,lb=bound if equality else None,name=name)
        rows[row]=dict(zip(x[matrix.indices[begin:end]],matrix.data[begin:end]))
    model.add(rows.keys());model.update()
    for row,terms in rows.items(): row.set_linear_coefficients(terms)


def enable():
    lp.OptlangInterfacer.make_sparse_constraint_system=staticmethod(bulk_constraint_system)
