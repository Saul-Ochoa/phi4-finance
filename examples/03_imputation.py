"""Fill in NVDA from same-day AAPL and MSFT (Section 3.4, eqs. 9-10).

Trains on the 230 trading days before the test window and compares the phi^4
conditional mean with the paper's rescaled-mean baseline R (eq. 10).
"""
import numpy as np

from phi4finance import Phi4Model, Scaler, load_returns

TRAIN_DAYS, TEST_DAYS = 230, 20


def main():
    rets = load_returns(["AAPL", "MSFT", "NVDA"], period="2y")
    train, test = rets.iloc[-(TRAIN_DAYS + TEST_DAYS):-TEST_DAYS], rets.iloc[-TEST_DAYS:]
    scaler = Scaler("absmax").fit(train)

    model = Phi4Model(n_stocks=3, lr=0.01, mu_global=False, lam_global=False, seed=0)
    model.fit(scaler.transform(train).to_numpy(), epochs=400, mcmc_steps=400)

    sd = train.std()
    phi4, base = [], []
    for _, row in test.iterrows():
        s = model.predict_conditional({0: row["AAPL"], 1: row["MSFT"]}, target_idx=2,
                                      n_samples=2000, burn=200, scaler=scaler)
        phi4.append(s.mean())
        base.append(sd["NVDA"] / 2 * (row["AAPL"] / sd["AAPL"] + row["MSFT"] / sd["MSFT"]))
    y = test["NVDA"].to_numpy()
    print(f"MAE phi4 = {np.abs(np.array(phi4) - y).mean():.4f}")
    print(f"MAE R    = {np.abs(np.array(base) - y).mean():.4f}")
    print(f"MAE zero = {np.abs(y).mean():.4f}")


if __name__ == "__main__":
    main()
