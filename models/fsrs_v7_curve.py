"""FSRS-7 curve candidates: FSRS-7 plus one parameter each (fsrs_curve.py).

The FSRS-7 recurrence is unchanged except for the added parameters. With them at 0 the
model is FSRS-7.

- again: the long-term stability seen by the forgetting curve is scaled by the card's
  Again count, S_eff = S * (1 + a)^(-w_again), where a is the number of Again answers
  among the card's earlier reviews. It applies wherever the curve is evaluated: the
  retrievability that drives the stability and difficulty updates, and the prediction.
- sibling: a new card's initial difficulty is blended with its siblings' current
  difficulty, D0 = (1 - b) * D0_FSRS-7(rating) + b * D_sib, b = w_sib in [0, 1]. D_sib
  and a has-siblings flag are input columns 2 and 3 of the card's history tensor,
  computed by fsrs_curve.py from reviews before the card's first review only.
"""

from typing import ClassVar

import torch
from torch import Tensor

from config import Config
from models.fsrs_v7 import PENALTY_W_L2, FSRS7, FSRS7ParameterClipper


def curve_model(again: bool, sibling: bool) -> type[FSRS7]:
    """FSRS-7 with the chosen extra parameters (indices 34, 35 in this order)."""
    extra = [("again", -0.5, 2.0)] * again + [("sibling", 0.0, 1.0)] * sibling
    index = {name: 34 + i for i, (name, _, _) in enumerate(extra)}

    class Clipper(FSRS7ParameterClipper):
        _CLIP_LO: ClassVar[list[float]] = FSRS7ParameterClipper._CLIP_LO + [
            lo for _, lo, _ in extra
        ]
        _CLIP_HI: ClassVar[list[float]] = FSRS7ParameterClipper._CLIP_HI + [
            hi for _, _, hi in extra
        ]

    class FSRS7Curve(FSRS7):
        init_w: ClassVar[list[float]] = FSRS7.init_w + [0.0] * len(extra)

        def __init__(self, config: Config, w: list[float] | None = None):
            super().__init__(config, w)
            self.clipper = Clipper(config)
            self._l2_sigma = torch.cat(
                [self._l2_sigma, torch.full((len(extra),), 0.5, device=self.config.device)]
            )
            self._again: Tensor | None = None  # ln(1 + Again count) for this step
            self._sib: tuple[Tensor, Tensor] | None = None  # (D_sib, has) at step 0

        # pyrefly: ignore[bad-override]
        def forgetting_curve(
            self, t: Tensor, s: Tensor, s_short: Tensor, d: Tensor
        ) -> Tensor:
            if self._again is not None:
                s = s * torch.exp(-self.w[index["again"]] * self._again)
            return super().forgetting_curve(t, s, s_short, d)

        # pyrefly: ignore[bad-override]
        def init_d(self, rating):
            d0 = super().init_d(rating)
            # mean reversion calls init_d(4) with an int: never blended
            if self._sib is None or not torch.is_tensor(rating):
                return d0
            d_sib, has = self._sib
            b = self.w[index["sibling"]]
            return torch.where(has > 0, (1 - b) * d0 + b * d_sib, d0)

        def _again_before(self, inputs: Tensor) -> Tensor | None:
            if not again:
                return None
            a = (inputs[:, :, 1] == 1).float()
            return torch.log1p(torch.cumsum(a, dim=0) - a)

        def forward(
            self, inputs: Tensor, state: Tensor | None = None
        ) -> tuple[Tensor, Tensor]:
            if state is None:
                state = torch.zeros((inputs.shape[1], 3), device=self.config.device)
            before = self._again_before(inputs)
            outputs = []
            for t, X in enumerate(inputs):
                self._again = None if before is None else before[t]
                self._sib = (X[:, 2], X[:, 3]) if sibling and t == 0 else None
                state = self.step(X, state)
                outputs.append(state)
            self._again, self._sib = None, None
            return torch.stack(outputs), state

        def batch_process(
            self,
            sequences: Tensor,
            delta_ts: Tensor,
            seq_lens: Tensor,
            real_batch_size: int,
        ) -> dict[str, Tensor]:
            outputs, _ = self.forward(sequences)
            rows = torch.arange(real_batch_size, device=self.config.device)
            final = outputs[seq_lens - 1, rows]
            if again:
                count = torch.cumsum((sequences[:, :, 1] == 1).float(), dim=0)
                self._again = torch.log1p(count[seq_lens - 1, rows])
            retentions = self.forgetting_curve(
                delta_ts, final[:, 0], final[:, 1], final[:, 2]
            )
            self._again = None
            L2_penalty = torch.sum(
                torch.square(self.w - self.init_w_tensor) / torch.square(self._l2_sigma)
            )
            return {
                "retentions": retentions,
                "stabilities": final[:, 0],
                "difficulties": final[:, 2],
                "penalty": PENALTY_W_L2 * L2_penalty * real_batch_size,
            }

    return FSRS7Curve


FSRS7Again = curve_model(again=True, sibling=False)
FSRS7Sibling = curve_model(again=False, sibling=True)
