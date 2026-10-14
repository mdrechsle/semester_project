#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from dataclasses import dataclass


# In[ ]:


@dataclass
class SinkhornNewtonConfig:
    epsilon : float = 1e-3
    max_iter : int = 100
    tol : float = 1e-10
    inner_tol : float = 1e-12
    inner_maxiter : int = 1000

    use_preconditionner : bool = True

    verbose : bool = False
    log_every : int = 1


# In[ ]:


@dataclass   
class SinkhornNewtonResult:
    P : np.array
    f : np.array
    g : np.array
    iterations : int
    converged : bool
    history : list[dict[str, Any]] = field(default_factory = list)


# In[ ]:


def compute_cost_matrix(x, y, cost_fn):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    C = np.asarray(cost_fn(x,y))
    
    return C


# In[ ]:


class SinkhornNewton:
    def __init__(self, laplacian_solver, config = None):
        self.solver = laplacian_solver
        self.config = config or SinkhornNewtonConfig()
        self.history = []


    def build_newton_system(self, P, epsilon):  #Builds the jacobian associated to the transport plan
        n, m = P.shape
        a_k = P.sum(axis = 1)
        b_k = P.sum(axis = 0)

        J = np.zeros((n+m, n+m))
        J[:n, :n] = np.diag(a_k) #Top left block
        J[:n, n:] = P            #Top right block 
        J[n:, :n] = P.T          #Bottom left block 
        J[n:, n:] = np.diag(b_k) #Bottom right block 

        J = J / epsilon

        return J, a_k, b_k

            

    def solve_newton_system(self, J, rhs, delta_f0 = None, delta_g0 = None):
        n = J.shape[0] // 2
        m = J.shape[0] - n

        ones_n = np.ones(n)
        ones_m = np.ones(m)
        kernel = np.concatenate([ones_n, -ones_m])

        kernel_norm_sq = n + m
        rhs_proj = rhs - (np.dot(kernel, rhs) / kernel_norm_sq) * kernel 

        x0 = None
        if delta_f0 is not None and delta_g0 is not None:
            x0 = np.concatenate([delta_f0, delta_g0])

        x, info = self.solver.solve(J, rhs_proj, x0 = x0, tol = self.config.inner_tol, maxiter = self.config.inner_maxiter)

        delta_f = x[:n]
        delta_g = x[n:]

        mean_f = np.mean(delta_f)
        mean_g = np.mean(delta_g)
        shift = (mean_f - mean_g) / 2
        delta_f = delta_f - shift
        delta_g = delta_g + shift

        return delta_f, delta_g, info

        
    def solve(self, a, b, C, epsilon = None, P0 = None):
        n, m = C.shape
        eps = epsilon if epsilon is not None else self.config.epsilon

        a = a / a.sum()
        b = b / b.sum()

        if P0 is not None:
            P = P0.copy()
        else:
            K = np.exp(-C / eps)
            P = K.copy()

        f = np.zeros(n)
        g = np.zeros(m)

        self.history = []
        converged = False

        for k in range(self.config.max_iter):
            a_k = P.sum(axis = 1)
            b_k = P.sum(axis = 0)

            viol_a = np.max(np.abs(a_k - a))
            viol_b = np.max(np.abs(b_k - b))
            max_viol = max(viol_a, viol_b)

            if self.config.verbose and k % self.config.log_every == 0:
                print(f"Iter {k}: max_viol = {max_viol:.2e}, "
                      f"solver_iter = {len(self.history[-1]['solver_info'].get('iterations', [0])) if self.history else 0}")
                
            if max_viol < self.config.tol:
                converged = True
                if self.config.verbose:
                    print(f"Converged at iteration {k}")
                break

                
            J, a_k, b_k = self.build_newton_system(P, eps)

            rhs = np.concatenate([a_k - a, b_k - b])
            delta_f, delta_g, solver_info = self.solve_newton_system(J, rhs)

            exp_df = np.exp(-delta_f / eps)
            exp_dg = np.exp(-delta_g / eps)

            P = np.diag(exp_df) @ K @ np.diag(exp_dg)

            f = f + delta_f
            g = g + delta_g

            self.history.append({
            'iteration' : k ,
            'viol_a' : viol_a,
            'viol_b' : viol_b,
            'max_viol' : max_viol, 
            'solver_info' : solver_info, 
            'norm_delta_f' : np.linalg.norm(delta_f),
            'norm_delta_g' : np.linalg.norm(delta_g)
                })

        return SinkhornNewtonResult(P=P, f=f, g=g, iterations = k + 1, converged = converged, history = self.history)

