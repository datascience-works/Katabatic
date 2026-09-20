from tabebm.TabEBM import TabEBM
import numpy as np

X = np.random.randn(50, 4)
y = np.random.randint(0, 2, 50)

tabebm = TabEBM()
data = tabebm.generate(X, y, num_samples=10)
print('Success! Generated classes:', list(data.keys()))
