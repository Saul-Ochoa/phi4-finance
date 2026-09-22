
from phi4finance.data import load_returns
from phi4finance.model import Phi4Model
tickers = ["ABT","AMGN","AXP","BAC","BMY","CAT","CL","COP","CVX","DE","DHR","EMR","EXC","GE","HD","HPQ","INTC","MSFT","ORCL","AAPL"]
rets = load_returns(tickers, period="10y")
print(rets.head())
model = Phi4Model(n_stocks=len(tickers), lr=0.01, mu_global=True, lam_global=True)
model.fit(rets.values, epochs=400, mcmc_steps=1500)
samples = model.predict_conditional({0:0.01,1:-0.01}, target_idx=19, n_samples=2000)
print(f"AAPL conditional mean={samples.mean():.4f} std={samples.std():.4f}")
