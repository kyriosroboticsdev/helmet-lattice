"""
stopping.py
===========
Stopping controller for the active-learning design loop.

Blends three criteria; the loop STOPS when ANY fires:

  (1) HARD CAP        -- max_iterations reached. Safety backstop; never infinite.
                         After a hard-cap stop you can manual-check and RESUME
                         (state is JSON-serializable) without losing history.
  (2) NO IMPROVEMENT  -- the best objective (Pareto hypervolume, or a headline
                         metric) has improved < rel_tol for `patience` iterations
                         in a row  ->  diminishing returns.
  (3) LOW EXPECTED     -- the optimizer's own qEHVI acquisition value for its best
      IMPROVEMENT       proposed candidate falls below `ei_floor` (as a fraction of
                        the first iteration's value) -> the model itself no longer
                        expects a meaningfully better design.

Design intent:
  * The HARD CAP is the guaranteed safety net (set generously).
  * (2) and (3) are the *smart* early-stops that halt when the search is actually
    done, usually well before the cap.
  * (3) is the most principled; (2) is a robust backup in case the surrogate is
    over-confident; (1) protects compute/deadline no matter what.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json, math
from pathlib import Path


@dataclass
class StopConfig:
    max_iterations: int = 60      # (1) HARD CAP  -- generous safety backstop
    patience: int = 5            # (2) stop after this many low-improvement rounds
    rel_tol: float = 0.01        # (2) "improvement" must beat best by > 1% to count
    ei_floor_frac: float = 0.05  # (3) stop when EI < 5% of the first iteration's EI
    min_iterations: int = 6      # never stop (except errors) before this many rounds


@dataclass
class StopState:
    iteration: int = 0
    best_metric: float = -math.inf      # higher = better (hypervolume or -injury)
    first_ei: float | None = None       # EI at the first iteration (for the fraction)
    stale_rounds: int = 0               # consecutive low-improvement rounds
    history: list = field(default_factory=list)   # per-iteration record
    stopped: bool = False
    stop_reason: str | None = None

    # -- persistence so a hard-cap stop can be manually reviewed then RESUMED --
    def save(self, path: str):
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))
    @classmethod
    def load(cls, path: str):
        d = json.loads(Path(path).read_text()); return cls(**d)


class StoppingController:
    """Call `update(...)` once per active-learning iteration. It records the round
    and returns (stop: bool, reason: str|None). Resume by loading a saved StopState
    and passing resume=True (which clears a prior hard-cap stop)."""

    def __init__(self, config: StopConfig | None = None, state: StopState | None = None):
        self.cfg = config or StopConfig()
        self.state = state or StopState()

    # ------------------------------------------------------------------
    def update(self, *, best_metric: float, top_ei: float,
               note: str = "") -> tuple[bool, str | None]:
        """
        best_metric : current best objective, higher = better. Use Pareto
                      hypervolume for the two-objective case, or a single headline
                      metric (e.g. best SEA). MUST be a 'bigger is better' number.
        top_ei      : the optimizer's acquisition (expected-improvement / qEHVI)
                      value for its BEST proposed candidate this round.
        """
        s, c = self.state, self.cfg
        s.iteration += 1
        if s.first_ei is None and top_ei is not None:
            s.first_ei = float(top_ei)

        # improvement bookkeeping (criterion 2)
        improved = best_metric > s.best_metric * (1 + c.rel_tol) if s.best_metric > -math.inf \
                   else True
        if improved:
            s.best_metric = float(best_metric); s.stale_rounds = 0
        else:
            s.stale_rounds += 1

        # expected-improvement fraction (criterion 3)
        ei_frac = (top_ei / s.first_ei) if (s.first_ei and s.first_ei > 0) else 1.0

        s.history.append(dict(
            iteration=s.iteration, best_metric=round(float(best_metric), 6),
            improved=bool(improved), stale_rounds=s.stale_rounds,
            top_ei=(None if top_ei is None else round(float(top_ei), 6)),
            ei_frac=round(float(ei_frac), 4), note=note))

        # ---- decide (respect a warm-up minimum first) ----
        stop, reason = False, None
        if s.iteration >= c.min_iterations:
            if s.stale_rounds >= c.patience:
                stop, reason = True, (f"NO_IMPROVEMENT: best metric improved < "
                                      f"{c.rel_tol*100:.0f}% for {c.patience} rounds")
            elif ei_frac < c.ei_floor_frac:
                stop, reason = True, (f"LOW_EXPECTED_IMPROVEMENT: EI fell to "
                                      f"{ei_frac*100:.1f}% of initial (< {c.ei_floor_frac*100:.0f}%)")
        # hard cap always checked, overrides warm-up
        if s.iteration >= c.max_iterations:
            stop, reason = True, (f"HARD_CAP: reached max_iterations={c.max_iterations} "
                                  f"(manual review then resume=True to continue)")

        s.stopped, s.stop_reason = stop, reason
        return stop, reason

    # ------------------------------------------------------------------
    def resume(self, extra_iterations: int | None = None):
        """After a HARD_CAP stop and a manual check, clear the stop and (optionally)
        extend the cap so the loop can continue."""
        if self.state.stop_reason and self.state.stop_reason.startswith("HARD_CAP"):
            if extra_iterations:
                self.cfg.max_iterations = self.state.iteration + extra_iterations
            self.state.stopped = False; self.state.stop_reason = None
            return True
        return False

    def summary(self) -> str:
        s = self.state
        lines = [f"iterations run : {s.iteration}",
                 f"best metric    : {s.best_metric:.6g}",
                 f"stopped        : {s.stopped}  ({s.stop_reason or '-'})"]
        return "\n".join(lines)


# ---- helper: Pareto hypervolume as the 'best_metric' for the 2-objective case ----
def pareto_hypervolume(df, objectives, ref_point=None) -> float:
    """DEPRECATED -- do not use for the FEA loop. Use fea_optimize.hypervolume.

    Two reasons: (1) it assumes EVERY objective is minimized, which silently treats
    higher SEA as worse (SEA is maximized); (2) its reference point defaults to the
    worst observed point, so it moves as data arrives and successive values are not
    comparable -- exactly what a no-improvement stopping rule must compare.
    fea_optimize.hypervolume takes per-objective signs and a reference fixed once per
    run."""
    raise NotImplementedError(
        "pareto_hypervolume assumes all-minimized objectives and a moving reference; "
        "use fea_optimize.hypervolume(rows, ref) instead.")


def _pareto_hypervolume_legacy(df, objectives, ref_point=None) -> float:
    """Original all-minimized implementation, kept only for reference."""
    import numpy as np
    Y = df[objectives].to_numpy(dtype=float)
    Y = Y[~np.isnan(Y).any(axis=1)]
    if len(Y) == 0: return -math.inf
    try:
        import torch
        from botorch.utils.multi_objective.hypervolume import Hypervolume
        if ref_point is None:
            ref = (Y.max(axis=0) * 1.05)
        else:
            ref = np.asarray(ref_point, float)
        hv = Hypervolume(ref_point=torch.as_tensor(-ref))
        from botorch.utils.multi_objective.pareto import is_non_dominated
        Yt = torch.as_tensor(-Y)
        front = Yt[is_non_dominated(Yt)]
        return float(hv.compute(front))
    except Exception:
        # fallback: simple 2D hypervolume vs ref (no torch)
        ref = (Y.max(axis=0) * 1.05) if ref_point is None else np.asarray(ref_point, float)
        pts = Y[np.argsort(Y[:, 0])]
        hv, prev_x = 0.0, ref[0]
        best_y = ref[1]
        for x, y in pts:
            if y < best_y:
                hv += (prev_x - x) * (ref[1] - y); best_y = y; prev_x = x
        return hv


if __name__ == "__main__":
    # tiny self-test: simulate a search that improves then plateaus
    import random
    ctrl = StoppingController(StopConfig(max_iterations=20, patience=4, min_iterations=3))
    best = 0.0
    for i in range(30):
        # improve fast early, then flatten
        gain = max(0.0, 0.3 * math.exp(-i/3))
        best += gain
        ei = 100 * math.exp(-i/4)                    # EI decays over time
        stop, why = ctrl.update(best_metric=best, top_ei=ei, note=f"gain={gain:.3f}")
        print(f"iter {i+1:2d}  best={best:.3f}  ei={ei:6.1f}  -> stop={stop} {why or ''}")
        if stop: break
    print("\n" + ctrl.summary())
