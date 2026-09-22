"""Next-day forecast of AAPL from its own history (Section 3.5, Appendix A.3).

Paper set-up: each configuration is 150 consecutive returns, the model is
trained on 80 configurations (230 trading days), and it then forecasts the
following 10 trading days from the 149 preceding returns. The scaler only
sees the training window. The zero forecast is reported as the baseline any
daily-return model must beat.

Runtime: a few minutes with the pure-Python sampler of v0.2 (V = 150).
"""
import numpy as np

from phi4finance import Phi4Model, Scaler, lag_embed, load_returns

WINDOW, N_TRAIN, HORIZON = 150, 80, 10


def main():
    r = load_returns("AAPL", period="3y", log=False)["AAPL"].to_numpy()
    split = len(r) - HORIZON                        # last 10 days are out of sample
    train_series = r[split - (N_TRAIN + WINDOW - 1):split]

    scaler = Scaler("absmax").fit(train_series)
    X = lag_embed(scaler.transform(train_series), WINDOW)
    assert X.shape == (N_TRAIN, WINDOW)

    model = Phi4Model(n_stocks=WINDOW, lr=0.005, mu_global=True, lam_global=True, seed=0)
    model.fit(X, epochs=200, mcmc_steps=300)

    preds, truth = [], []
    for t in range(split, len(r)):
        history = r[t - (WINDOW - 1):t]             # 149 returns before day t
        s = model.forecast_next_day(history, n_samples=2000, burn=200, scaler=scaler)
        preds.append(s.mean()); truth.append(r[t])
    preds, truth = np.array(preds), np.array(truth)
    print(f"MAE phi4 = {np.abs(preds - truth).mean():.4%}")
    print(f"MAE zero = {np.abs(truth).mean():.4%}")
    print(f"direction hit rate = {np.mean(np.sign(preds) == np.sign(truth)):.0%}")


if __name__ == "__main__":
    main()
