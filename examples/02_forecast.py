
from phi4finance.data import load_returns
from phi4finance.model import Phi4Model
import numpy as np
rets = load_returns(["AAPL"], period="2y")
aapl = rets["AAPL"].values
window=150
X_full=np.array([aapl[i:i+window+1] for i in range(len(aapl)-window-1)])
model=Phi4Model(n_stocks=window+1, lr=0.005)
model.fit(X_full, epochs=300, mcmc_steps=1200)
history=aapl[-window-1:-1]
pred=model.predict_conditional({i:history[i] for i in range(window)}, target_idx=window, n_samples=3000)
print(f"Real: {aapl[-1]:.4f} | Pred mean: {pred.mean():.4f}")
