#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.sparse.linalg import cg, LinearOperator, spilu
from scipy.sparse import csr_matrix
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import warnings 
from mwst import mwsts
from mwst import preconditioner
import apxchol
import scipy.sparse as sp
import time



# In[6]:


class LaplacianSolver(ABC):
    
    @abstractmethod
    def solve(self, L, b, x0 = None, tol = 1e-10, maxiter = 1000):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @staticmethod
    def _project_kernel(b):
        n = b.shape[0]
        return b - (np.sum(b) / n) * np.ones(n)

    @staticmethod
    def _project_solution(x):
        n = x.shape[0]
        return x - (np.sum(x) / n) * np.ones(n)


# In[7]:


class ConjugateGradientSolver(LaplacianSolver):

    def __init__(self, preconditioner = None, mwst_n_trees=10, mwst_oversample=8, mwst_floor = 1e-14):
        valid = (None, 'jacobi', 'incomplete_cholesky', 'apxchol')
        if preconditioner not in valid:
            raise ValueError(f"Unknown preconditionner : {preconditioner}. Valid : {valid}")
        self.preconditioner = preconditioner
        self.mwst_n_trees = mwst_n_trees
        self.mwst_oversample = mwst_oversample
        self.mwst_floor = mwst_floor 
        self.last_setup_time = 0.0
        self._name = f"CG({preconditioner or 'none'})"

    @property
    def name(self) -> str:
        return self._name

    def solve(self, L, b, x0=None, tol=1e-10, maxiter=1000, kernel=None):
        n = L.shape[0]
        if kernel is None:
            kernel = np.ones(n) / np.sqrt(n)
        kernel = kernel / np.linalg.norm(kernel)
    
        b_proj = b - np.dot(kernel, b) * kernel
        b_norm = np.linalg.norm(b_proj)
        if b_norm < 1e-300:
            return np.zeros(n), {
                'iterations': 0, 'residual': 0.0, 'converged': True,
                'solver': self.name, 'note': 'b in kernel, x = 0',
            }
    
        if x0 is not None:
            x0 = x0 - np.dot(kernel, x0) * kernel
    
        t0 = time.perf_counter()
        M = self._build_preconditioner(L, kernel)
        self.last_setup_time = time.perf_counter() - t0
    
        iter_count = [0]
        def callback(xk):
            iter_count[0] += 1
    
        x, info = cg(L, b_proj, x0=x0, rtol=tol, maxiter=maxiter, M=M, callback=callback)
    
        x = x - np.dot(kernel, x) * kernel
    
        residual = np.linalg.norm(L @ x - b_proj) / (b_norm + 1e-300)
    
        return x, {
            'iterations': int(iter_count[0]),
            'residual': float(residual),
            'converged': bool(info == 0),
            'solver': self.name,
            'scipy_info': int(info),
            'setup_time': self.last_setup_time
        }

    def _build_preconditioner(self, L, kernel = None):
        n = L.shape[0]

        if self.preconditioner is None:
            return None
            
        if self.preconditioner == 'jacobi':
            diag = np.diag(L).astype(float).copy()
            diag[np.abs(diag) < 1e-300] = 1.0
            return LinearOperator((n, n), matvec=lambda v: v / diag)

        if self.preconditioner == 'incomplete_cholesky':
            try:
                Lc = sp.csc_matrix(L)
                scale = float(abs(Lc).max()) if Lc.nnz else 1.0
                shift = 1e-10 * max(1.0, scale)
                ilu = spilu(Lc + shift * sp.identity(n, format='csc'))
                return LinearOperator((n, n), matvec=ilu.solve, dtype=float)
            except Exception as e:
                warnings.warn(f"Preconditioner 'incomplete_cholesky' unavailable "
                              f"({type(e).__name__}: {e}). CG without preconditioner.")
                return None
                
        if self.preconditioner == 'apxchol':
            return self._build_apxchol(L, kernel)
                
        return None

    def _build_apxchol(self, L, kernel):

        N = L.shape[0]
        if kernel is None:
            warnings.warn("apxchol: kernel (sign pattern) required. CG without preconditioner.")
            return None
        sgn = np.where(kernel >= 0, 1.0, -1.0)
        n = int(np.sum(sgn > 0))
        if 2 * n != N or not (np.all(sgn[:n] > 0) and np.all(sgn[n:] < 0)):
            warnings.warn("apxchol: needs a square bipartite problem (n == m) with kernel "
                          "[1,..,1,-1,..,-1]. CG without preconditioner.")
            return None
 
        Ld = L.toarray() if sp.issparse(L) else np.asarray(L)
        W = np.array(Ld[:n, n:], dtype=np.float64, order='C')
 
        # ---- sanity checks: this must really be S*Laplacian*S with a positive weight block
        dmax = max(float(np.abs(np.diag(Ld)).max()), 1e-300)
        ok = (W.min() >= 0
              and np.allclose(np.diag(Ld)[:n], W.sum(axis=1), rtol=1e-6, atol=1e-10 * dmax)
              and np.allclose(np.diag(Ld)[n:], W.sum(axis=0), rtol=1e-6, atol=1e-10 * dmax))
        if not ok:
            warnings.warn("apxchol: matrix is not of the form S*Laplacian*S (weights<0 or "
                          "degrees != row sums). CG without preconditioner.")
            return None
 
        # Make sure that the graph is connected : a non-existing edge becomes an edge with mwst-floor weight
        floor = self.mwst_floor * float(W.max())
        W = np.maximum(W, floor)
 
        try:
            edges = mwsts(W, self.mwst_n_trees, self.mwst_oversample)
            M_red = preconditioner(W, edges)      # apxchol on the grounded tree-union Laplacian
        except Exception as e:
            warnings.warn(f"Preconditioner 'apxchol' unavailable ({type(e).__name__}: {e}). "
                          f"CG without preconditioner.")
            return None
 
        apply_red = M_red.matvec if hasattr(M_red, 'matvec') else (lambda v: M_red @ v)
 
        def matvec(v):
            u = sgn * np.asarray(v, dtype=float).ravel()   # to Laplacian space
            u = u - u.mean()                               # project on range(L)
            z = np.zeros(N)
            z[:-1] = apply_red(u[:-1])                     # grounded solve
            z -= z.mean()                                  # = pseudo-inverse of the sparsifier
            return sgn * z                                 # back to signless space
 
        return LinearOperator((N, N), matvec=matvec, rmatvec=matvec, dtype=float)