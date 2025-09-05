import numpy as np
from katabatic.models.meg import MEG

X = np.random.randint(0, 10, size=(500, 5))

G = MEG(epochs=2)   # small epochs for quick test
G.fit(X)
S = G.sample(10)
print("Synthetic shape:", S.shape)
print(S)
