#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix


# In[5]:


from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
from scipy.sparse.linalg import spilu
import warnings 


# In[6]:


class LaplacianSolver(ABC):
    
    @abstractmethod
    def solve(self, L, b, x0, tol, maxiter):
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

    def __init__(self, preconditioner = None):
        if preconditioner not in (None, 'jacobi', 'incomplete_cholesky'):
            raise ValueError("Unknown preconditionner")
        self.preconditioner = preconditioner
        self.name = f"CG({preconditioner or 'none'})"

    def name(self) -> str:
        return self._name
        

    def solve(self, L, b, x0 = None, tol = 1e-10, maxiter = 1000):
        n = L.shape[0]

        b_proj = self._project_kernel(b)
        b_norm = np.linalg.norm(b_proj)
        if b_norm < 1e-300:
            return np.zeros(n), {
                'iterations': 0,
                'residual': 0.0,
                'converged': True,
                'solver': self.name,
                'note': 'b in kernel, x = 0',
            }

        if x0 is None:
            x0 = np.zeros(n)
        else:
            x0 = self._project_solution(x0)

        M = self._build_preconditioner(L)
        x, info = cg(L, b_proj, x0= x0, rtol = tol, maxiter = maxiter, M = M)
        residual_vec = L @ x - b_proj
        residual = np.linalg.norm(residual_vec) / (b_norm + 1e-300)

        x = self._project_solution(x)
        converged = (info == 0)

        return x, {'iterations': int(info) if info >= 0 else int(maxiter),
            'residual': float(residual),
            'converged': converged,
            'solver': self.name,
            'scipy_info': int(info),
        }

    def _build_preconditioner(self, L):
        n = L.shape[0]

        if self.preconditioner is None:
            return None
            
        if self.preconditioner == 'jacobi':
            diag = np.diag(L).astype(float).copy()
            diag[np.abs(diag) < 1e-300] = 1.0
            return LinearOperator((n, n), matvec=lambda v: v / diag)

        if self.preconditioner == 'incomplete_cholesky':
            try:
                shift = 1e-10 * max(1.0, np.abs(L).max())
                L_reg = csr_matrix(L + shift* np.eye(n)).tocsc()

                ilu = spilu(L_reg)
                return LinearOperator((n, n)matvec = ilu.solve)
            except Exception as e:
                warnings.warn(
                    f"Préconditionneur 'incomplete_cholesky' unavailable "
                    f"({type(e).__name__}: {e}). non-preconditionned CG."
                )
                return None
                
        return None

