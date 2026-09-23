import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

MODELS_RUN = {
    "CODI": {
        "module": "katabatic.models.codi.models",
        "class": "CODI",
        "params": {
            "n_steps": 50,
            "epochs": 100,
            "batch_size": 256,
        },
    },
    "CTGAN": {
        "module": "katabatic.models.ctgan.models",
        "class": "CTGANModel",
        "params": {
            "epochs": 100,
            "batch_size": 256,
            "seed": 42,
        },
    },
    "NAIVEBAYES": {
        "module": "katabatic.models.naivebayes.models",
        "class": "NaiveBayesModel",
        "params": {
            "seed": 42,
        },
    },
    "GANBLR": {
        "module": "katabatic.models.ganblr.models",
        "class": "GANBLR",
        "params": {},
    },
}
