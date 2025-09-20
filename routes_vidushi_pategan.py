# routes_vidushi_pategan.py
from fastapi import APIRouter
import pandas as pd
from Models import MODEL_REGISTRY

router = APIRouter()

@router.get("/vidushi_pategan/generate")
def generate_pategan(n: int = 10):
    """Generate synthetic samples using Vidushi_PATEGAN"""
    model = MODEL_REGISTRY["Vidushi_PATEGAN"]()
    
    # For demo: train on Iris
    from sklearn.datasets import load_iris
    X, y = load_iris(as_frame=True, return_X_y=True)
    X["target"] = y
    model.train(X, y)

    df = model.generate(n)
    return {"data": df.to_dict(orient="records")}
