"""Same frozen multinomial family and structural/static encoding; new refi block."""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from credit_risk.track_b.macro_hazard.data import MACRO
from credit_risk.track_b.macro_hazard.models import Encoder
from credit_risk.track_b.macro_hazard.protocol import NUMERIC

from .features import representation

CONTEXT = ("unemployment_level", "unemployment_change_3m", "hpi_yoy", "cpi_yoy", "gdp_qoq")


@dataclass
class RefiEncoder(Encoder):
    refi: bool = False
    linear: bool = False
    context: bool = False

    def raw_numeric(self, data):
        pieces = [super().raw_numeric(data)]
        if self.refi:
            pieces.append(representation(data, self.linear))
        if self.context:
            values = data["macro"][:, [MACRO.index(n) for n in CONTEXT]]
            if not np.isfinite(values).all():
                raise ValueError("Missing frozen non-rate context")
            pieces.append(values)
        return np.column_stack(pieces)

    def fit(self, data):
        if self.macro or (self.context and not self.refi) or (self.refi and not self.mortgage):
            raise ValueError("Invalid/redundant refi ladder terms")
        super().fit(data)
        extra = (["REFI_GAP"] if self.linear else ["REFI_POS", "REFI_NEG"]) if self.refi else []
        if self.context:
            extra += list(CONTEXT)
        at = len(NUMERIC) if self.mortgage else 0
        self.names[at:at] = extra
        return self


def fit(data, name):
    if name not in {"P0", "P1", "P2", "LINEAR", "P3"}:
        raise ValueError("Unprespecified model")
    if np.any(data["role"] != 0) or set(np.unique(data["event"])) != {0, 1, 2}:
        raise ValueError("Development-only fitting with all three competing events required")
    encoder = RefiEncoder(
        mortgage=name != "P0",
        refi=name not in {"P0", "P1"},
        linear=name == "LINEAR",
        context=name == "P3",
    ).fit(data)
    encoder.parameters["trained_vintages"] = np.unique(data["vintage"]).astype(int).tolist()
    model = LogisticRegression(C=1, solver="lbfgs", max_iter=3000, tol=1e-8, random_state=61010)
    with threadpool_limits(limits=1):
        model.fit(encoder.transform(data), data["event"])
    if model.n_iter_.max() >= 3000 or not np.isfinite(model.coef_).all():
        raise ValueError("STOP: prespecified fit failed; no retuning permitted")
    return encoder, model
