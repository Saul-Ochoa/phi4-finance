"""Fit a 20-stock phi^4 model and sample a conditional (Sections 3.1-3.2).

Stocks are the paper's list (Appendix B). The scaler is fitted on the
training window only; conditioning values are given in return units.
"""
import warnings

import numpy as np

from phi4finance import Phi4Model, Scaler, load_returns
from phi4finance.metrics import market_kurtosis

TICKERS = ["ABT", "AMGN", "AXP", "BAC", "BMY", "CAT", "CL", "COP", "CVX", "DE",
           "DHR", "EMR", "EXC", "GE", "HD", "HPQ", "INTC", "MSFT", "ORCL", "AAPL"]


def main():
    rets = load_returns(TICKERS, period="10y")
    train = rets.iloc[:-250]                      # hold out the last year
    scaler = Scaler("absmax").fit(train)
    X = scaler.transform(train).to_numpy()

    model = Phi4Model(n_stocks=len(TICKERS), mu_global=True, lam_global=True, seed=0)
    model.fit(X, method="pl")                      # pseudo-likelihood: seconds, no MCMC
    # optional likelihood refinement from the PL solution (the paper's estimator):
    # model.fit(X, method="ml", epochs=200, mcmc_steps=30)

    # p(AAPL | ABT = +1%, AMGN = -1%), other 17 stocks marginalised by MCMC
    aapl = model.predict_conditional({0: 0.01, 1: -0.01}, target_idx=19,
                                     n_samples=3000, burn=500, scaler=scaler)
    print(f"AAPL | ABT=+1%, AMGN=-1%: mean={aapl.mean():+.4%}  std={aapl.std():.4%}")

    # Higher-order statistics: data vs model vs binarised (Fig. 1 in miniature)
    sims = scaler.inverse_transform(model.sample(n_samples=3000, burn=500, thin=2))
    with warnings.catch_warnings():  # binarised days with all-equal signs have no kurtosis
        warnings.simplefilter("ignore", RuntimeWarning)
        k_bin = np.nanmean(market_kurtosis(np.sign(train)))
    print(f"mean cross-sectional kurtosis  data={market_kurtosis(train).mean():.2f}  "
          f"phi4={market_kurtosis(sims).mean():.2f}  binarised={k_bin:.2f}")

    iu = np.triu_indices(len(TICKERS), 1)
    corr = np.corrcoef(X.T)[iu]
    agree = np.mean(np.sign(model.W[iu]) == np.sign(corr))
    print(f"sign(w_ij) agrees with sign(corr_ij) on {agree:.0%} of pairs")


if __name__ == "__main__":
    main()
