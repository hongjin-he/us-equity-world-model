"""
§6.2 — Lyapunov regime detector (Theorem VII.1).

RiskIndex(t) = LV(z_t) / V(z_t)  =  |∇V(z_t)| / V(z_t)

When RiskIndex > threshold_crisis → market in unstable regime (bubble/crisis)
When RiskIndex < threshold_recovery → return to normal

Corollary VII.1: This is a core application of the world model beyond return prediction —
real-time monitoring of market stability.
"""
import torch


class LyapunovRegimeDetector:
    def __init__(
        self,
        threshold_crisis: float = 0.85,
        threshold_recovery: float = 0.40,
    ):
        self.thresh_crisis   = threshold_crisis
        self.thresh_recovery = threshold_recovery
        self.regime = "normal"
        self.history: list[dict] = []

    def lyapunov_indicator(self, z_t: torch.Tensor, V_model: torch.nn.Module) -> float:
        """
        Compute instability proxy: |∇V(z_t)| / V(z_t).
        Full Itô generator: LV = ∇V·m* + (σ²/2)·tr(∇²V) + ∫[V(z+y)-V(z)-∇V·y]ν^η(dy)
        Simplified here to dominant term for online monitoring.
        """
        z = z_t.detach().requires_grad_(True)
        t = torch.tensor([[0.5]])
        V = V_model(z.unsqueeze(0) if z.dim() == 1 else z, t).squeeze()
        dV = torch.autograd.grad(V, z)[0]
        instability = dV.norm().item() / (V.item() + 1e-6)
        return instability

    def update(self, z_t: torch.Tensor, V_model: torch.nn.Module) -> tuple[str, float]:
        indicator = self.lyapunov_indicator(z_t, V_model)
        prev_regime = self.regime

        if indicator > self.thresh_crisis and self.regime == "normal":
            self.regime = "crisis"
        elif indicator < self.thresh_recovery and self.regime == "crisis":
            self.regime = "recovery"
        elif self.regime == "recovery" and indicator < self.thresh_recovery * 0.8:
            self.regime = "normal"

        self.history.append({"regime": self.regime, "indicator": indicator})

        if self.regime != prev_regime:
            print(f"[REGIME] {prev_regime} → {self.regime}  (RiskIndex={indicator:.3f})")
            self._on_regime_change(self.regime, indicator)

        return self.regime, indicator

    def _on_regime_change(self, new_regime: str, indicator: float) -> None:
        if new_regime == "crisis":
            # Signal to Controller: halve leverage, tighten CVaR to 2.5%
            print("[ACTION] Halving leverage, tightening CVaR to 2.5%")
        elif new_regime == "recovery":
            print("[ACTION] Gradual leverage restoration")
