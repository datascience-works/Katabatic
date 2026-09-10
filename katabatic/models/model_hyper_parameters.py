import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from katabatic.models.codi.models import CODI  # noqa: E402
from katabatic.models.ctgan.models import CTGANModel  # noqa: E402
from katabatic.models.naivebayes.models import NaiveBayesModel  # noqa: E402
from katabatic.models.ganblr.models import GANBLR  # noqa: E402

MODELS = {
    "CODI": {
        "class": CODI,
        "params": {
            "n_steps": 50,
            "epochs": 100,
            "batch_size": 256,
        },
    },
    "CTGAN": {
        "class": CTGANModel,
        "params": {
            "epochs": 100,
            "batch_size": 256,
            "seed": 42,
        },
    },
    "NAIVEBAYES": {
        "class": NaiveBayesModel,
        "params": {
            "seed": 42,
        },
    },
    # Note: Hyper parameters for GANBLR are set within the models.py for GANBLR
    "GANBLR": {
        "class": GANBLR,
        "params": {
        },
    },          
}


# print(MODELS.keys())
# print(model)
