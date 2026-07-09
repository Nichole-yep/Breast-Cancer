
# AUTO PATH FIX FOR FINAL GITHUB STRUCTURE
from pathlib import Path as _Path
import sys as _sys
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
for _p in [_PROJECT_ROOT, _PROJECT_ROOT / "src"]:
    _s = str(_p)
    if _s not in _sys.path:
        _sys.path.insert(0, _s)
# END AUTO PATH FIX
# evaluate/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCETverskyLoss(nn.Module):
    def __init__(
        self,
        pos_weight=15.0,
        alpha=0.2,
        beta=0.8,
        smooth=1e-5
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight)
        )

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        TP = (probs * targets).sum()
        FP = (probs * (1 - targets)).sum()
        FN = ((1 - probs) * targets).sum()

        tversky = (TP + self.smooth) / (
            TP + self.alpha * FN + self.beta * FP + self.smooth
        )
        tversky_loss = 1.0 - tversky

        return 0.5 * bce_loss + 0.5 * tversky_loss