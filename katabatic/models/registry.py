from katabatic.models.meg import MEG

REGISTRY = {
    "meg": MEG,
}

def get_model(name: str):
    key = name.lower()
    if key not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(REGISTRY.keys())}")
    return REGISTRY[key]
