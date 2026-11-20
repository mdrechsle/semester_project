import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np
from scipy.special import logsumexp

from CG_Solver import LaplacianSolver 


@dataclass
class SinkhornNewtonConfig:
    epsilon: float = 1e-3          # target (final) epsilon
    max_iter: int = 100            # Newton iterations at the final epsilon
    tol: float = 1e-10             # max marginal violation at the final epsilon
    inner_tol: float = 1e-8        # relative tolerance of the linear solve (see notes)
    inner_maxiter: int = 1000

    sinkhorn_warmup_iters: int = 1    # log-domain Sinkhorn iterations at the start of each stage (0 = off)
    line_search: bool = True       # halve the Newton step until the dual objective decreases
    max_backtracks: int = 30

    verbose: bool = False
    log_every: int = 1

    use_annealing: bool = True
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.5
    anneal_tol: float = 1e-6
    anneal_max_iter: int = 100
    min_epsilon: float = 1e-8


@dataclass
class SinkhornNewtonResult:
    P: np.ndarray
    f: np.ndarray
    g: np.ndarray
    iterations: int
    converged: bool
    history: list = field(default_factory=list)
    delta_f: Optional[np.ndarray] = None
    delta_g: Optional[np.ndarray] = None

    def marginal_error(self, a, b):
        a_k = self.P.sum(axis=1)
        b_k = self.P.sum(axis=0)
        return (
            float(np.max(np.abs(a_k - a))),
            float(np.max(np.abs(b_k - b))),
        )


def compute_cost_matrix(x, y, cost_fn):
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    return np.asarray(cost_fn(x, y))


class SinkhornNewton:
    """
    Convention: P_ij = exp(-(f_i + g_j + C_ij) / eps).
    Dual objective (to minimise): Phi(f, g) = <a, f> + <b, g> + eps * sum(P)
        grad_f = a - P 1,  grad_g = b - P^T 1
        Hessian = [[diag(P1), P], [P^T, diag(P^T 1)]] / eps  = J
    Newton step: J delta = (P1 - a, P^T 1 - b), then f += t*delta_f, g += t*delta_g.
    """

    def __init__(self, laplacian_solver, config=None):
        self.solver = laplacian_solver
        self.config = config or SinkhornNewtonConfig()

    # ------------------------------------------------------------------ plan
    @staticmethod
    def _plan(f, g, C, eps):
        """P = exp(-(f + g + C)/eps), computed with a single exponent (no under/overflow
        from separate exp(-C/eps), exp(-f/eps), exp(-g/eps) factors)."""
        M = C + f[:, None]
        M += g[None, :]
        M /= -eps
        np.exp(M, out=M)
        return M

    @staticmethod
    def _shift_to_unit_mass(f, g, C, eps):
        """Return f' = f + c (constant c) such that sum(P(f', g)) = 1, computed stably."""
        M = C + f[:, None]
        M += g[None, :]
        m = M.min()
        Z = np.exp(-(M - m) / eps).sum()
        return f + (-m + eps * np.log(Z))

    # --------------------------------------------------------------- newton
    def build_newton_system(self, P, epsilon):
        n, m = P.shape
        a_k = P.sum(axis=1)
        b_k = P.sum(axis=0)

        J = np.empty((n + m, n + m))
        J[:n, :n] = 0.0
        J[n:, n:] = 0.0
        J[:n, n:] = P
        J[n:, :n] = P.T
        J[np.arange(n), np.arange(n)] = a_k
        J[n + np.arange(m), n + np.arange(m)] = b_k
        J /= epsilon
        return J, a_k, b_k

    def solve_newton_system(self, J, rhs, n):
        N = J.shape[0]
        m = N - n

        kernel = np.concatenate([np.ones(n), -np.ones(m)])
        kernel /= np.linalg.norm(kernel)

        comp = float(np.dot(kernel, rhs))
        rhs_norm = float(np.linalg.norm(rhs))
        if abs(comp) > 1e-8 * rhs_norm + 1e-13:  
            warnings.warn(f"Newton rhs not orthogonal to kernel: <k,rhs>={comp:.2e}, "
                          f"|rhs|={rhs_norm:.2e} (are a and b normalised to the same mass?)")
        rhs_proj = rhs - comp * kernel

        x, info = self.solver.solve(
            J, rhs_proj,
            x0=None,
            tol=self.config.inner_tol,
            maxiter=self.config.inner_maxiter,
            kernel=kernel,
        )
        x = x - np.dot(kernel, x) * kernel
        return x[:n], x[n:], info

    # ------------------------------------------------------------ annealing
    def _build_epsilon_schedule(self, eps_target):
        if not self.config.use_annealing:
            return [eps_target]
        eps_start = self.config.epsilon_start
        if eps_start <= eps_target:
            return [eps_target]
        schedule = []
        eps = eps_start
        while eps > eps_target:
            schedule.append(float(eps))
            eps *= self.config.epsilon_decay
            if eps < self.config.min_epsilon:
                break
        schedule.append(float(eps_target))
        return schedule

    # -------------------------------------------------------- one epsilon
    def _solve_single_epsilon(self, a, b, C, eps, f_init, g_init, tol, max_iter):
        cfg = self.config
        n, m = C.shape
        f, g = f_init.copy(), g_init.copy()
        history = []
        converged = False
        delta_f = np.zeros(n)
        delta_g = np.zeros(m)
        n_newton = 0

        for k in range(max_iter + 1):
            # P is rebuilt from the potentials at every iteration: no multiplicative drift
            P = self._plan(f, g, C, eps)
            a_k = P.sum(axis=1)
            b_k = P.sum(axis=0)
            viol_a = float(np.max(np.abs(a_k - a)))
            viol_b = float(np.max(np.abs(b_k - b)))
            max_viol = max(viol_a, viol_b)

            if cfg.verbose and (k % cfg.log_every == 0):
                si = history[-1]['solver_info'] if history else {}
                print(f"    iter {k}: max_viol = {max_viol:.3e}  "
                      f"cg_iters={si.get('iterations', '-')}  "
                      f"cg_res={si.get('residual', 0):.2e}  "
                      f"cg_conv={si.get('converged', '-')}")

            if max_viol < tol:
                converged = True
                break
            if k == max_iter:
                break

            if np.any(a_k <= 0) or np.any(b_k <= 0):
                warnings.warn("A row/column of P underflowed to 0: the Newton system is "
                              "inconsistent. Increase epsilon or use the annealing schedule.")
                break

            t0 = time.perf_counter()
            J, a_k, b_k = self.build_newton_system(P, eps)
            t_build = time.perf_counter() - t0
            rhs = np.concatenate([a_k - a, b_k - b])

            t0 = time.perf_counter()
            delta_f, delta_g, solver_info = self.solve_newton_system(J, rhs, n)
            t_solve = time.perf_counter() - t0
            del J

            # ---- Step-size safety: halve the step until the dual objective does not increase, so that the plan doesn't blow up
            t_step = 1.0
            if cfg.line_search:
                phi0 = float(a @ f + b @ g + eps * P.sum())
                for _ in range(cfg.max_backtracks + 1):
                    u = np.exp(np.clip(-t_step * delta_f / eps, -700.0, 700.0))
                    v = np.exp(np.clip(-t_step * delta_g / eps, -700.0, 700.0))
                    with np.errstate(over='ignore', invalid='ignore'):   # rejected trials may overflow
                        sumP = float(u @ (P @ v))
                    phi = float(a @ (f + t_step * delta_f) + b @ (g + t_step * delta_g) + eps * sumP)
                    if phi <= phi0 + 1e-12 * (1 + abs(phi0)):    # tolerance = round-off only
                        break
                    t_step *= 0.5
                else:
                    warnings.warn("Step-size search failed; stopping this stage.")
                    break

            f = f + t_step * delta_f
            g = g + t_step * delta_g
            n_newton += 1

            history.append({
                'iteration': k,
                'viol_a': viol_a,
                'viol_b': viol_b,
                'max_viol': max_viol,
                'solver_info': solver_info,
                'step_size': t_step,
                'time_build_system': t_build,
                'time_linear_solve': t_solve,
                'norm_delta_f': float(np.linalg.norm(delta_f)),
                'norm_delta_g': float(np.linalg.norm(delta_g)),
            })

        return f, g, converged, history, n_newton, delta_f, delta_g

    # ------------------------------------------------------- sinkhorn warm-up
    @staticmethod
    def _sinkhorn_warmup(a, b, C, eps, f, g, n_iter):
        """A few log-domain Sinkhorn iterations: cheap, robust, brings the marginals close to
        (a, b) so that the first Newton steps are not wild. Never over/underflows."""
        la = np.log(np.maximum(a, 1e-300))
        lb = np.log(np.maximum(b, 1e-300))
        for _ in range(n_iter):
            f = eps * logsumexp(-(g[None, :] + C) / eps, axis=1) - eps * la
            g = eps * logsumexp(-(f[:, None] + C) / eps, axis=0) - eps * lb
        return f, g

    # ------------------------------------------------------------------ main
    def solve(self, a, b, C, epsilon=None, f0=None, g0=None):
        n, m = C.shape
        eps_target = epsilon if epsilon is not None else self.config.epsilon
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        a = a / a.sum()
        b = b / b.sum()

        eps_schedule = self._build_epsilon_schedule(eps_target)
        if self.config.verbose:
            print(f"Annealing: {len(eps_schedule)} stages, "
                  f"{eps_schedule[0]:.3e} -> {eps_schedule[-1]:.3e}")

        f = np.asarray(f0, dtype=float).copy() if f0 is not None else -C.min(axis=1)
        g = np.asarray(g0, dtype=float).copy() if g0 is not None else np.zeros(m)

        full_history = []
        global_iter = 0
        all_converged = True
        delta_f = np.zeros(n)
        delta_g = np.zeros(m)

        for stage_idx, eps in enumerate(eps_schedule):
            is_last = (stage_idx == len(eps_schedule) - 1)
            stage_tol = self.config.tol if is_last else self.config.anneal_tol
            stage_max_iter = self.config.max_iter if is_last else self.config.anneal_max_iter

            f = self._shift_to_unit_mass(f, g, C, eps)
            if self.config.sinkhorn_warmup_iters > 0:
                f, g = self._sinkhorn_warmup(a, b, C, eps, f, g, self.config.sinkhorn_warmup_iters)

            if self.config.verbose:
                print(f"\n=== Stage {stage_idx+1}/{len(eps_schedule)} : "
                      f"eps = {eps:.3e}, tol = {stage_tol:.1e} ===")

            f, g, stage_conv, stage_hist, n_iter, delta_f, delta_g = \
                self._solve_single_epsilon(a, b, C, eps, f, g,
                                           tol=stage_tol, max_iter=stage_max_iter)

            if not stage_conv and is_last:
                all_converged = False

            for entry in stage_hist:
                entry['stage'] = stage_idx
                entry['epsilon'] = float(eps)
                entry['global_iteration'] = global_iter
                global_iter += 1
                full_history.append(entry)

            if self.config.verbose:
                status = "converged" if stage_conv else "NOT converged"
                print(f"  -> {status} in {n_iter} Newton iterations")

        P = self._plan(f, g, C, eps_schedule[-1])
        return SinkhornNewtonResult(
            P=P, f=f, g=g,
            iterations=global_iter,
            converged=all_converged,
            history=full_history,
            delta_f=delta_f,
            delta_g=delta_g,
        )