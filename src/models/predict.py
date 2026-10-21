
import os
from pathlib import Path

import skops.io as sio


def load_churn_model():
    project_root = Path(__file__).resolve().parents[2]

    model_path = os.getenv(
        "MODEL_PATH",
        str(project_root / "deployment" / "model" / "model.skops"),
    )

    trusted_types = [
        "_loss.CyHalfBinomialLoss",
        "sklearn.tree._tree.Tree",
    ]

    return sio.load(
        model_path,
        trusted=trusted_types,
    )
