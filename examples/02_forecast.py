"""Next-day forecast of AAPL from its own history (Section 3.5, Appendix A.3).

Paper set-up: each configuration is 150 consecutive returns, the model is
trained on 80 configurations (230 trading days) and then forecasts the
following 10 trading days from the 149 preceding returns. The scaler only
sees the training window.

v0.3: the model is fitted by pseudo-likelihood (seconds) and each forecast is
the exact 1-D conditional p(phi_1 | phi_0, ..., phi_-148), so the output is a
full predictive distribution: mean, interval and coverage are reported next
to the zero forecast, the baseline any daily-return model must beat.
"""
import numpy as np

from phi4finance import Phi4Model, Scaler, lag_embed, load_returns

WINDOW, N_TRAIN, HORIZON = 150, 80, 10
L2 = 0.5


def main():
    r = load_returns("AAPL", period="3y", log=False)["AAPL"].to_numpy()
    split = len(r) - HORIZON                        # last 10 days are out of sample
    train_series = r[split - (N_TRAIN + WINDOW - 1):split]

    scaler = Scaler("absmax").fit(train_series)
    X = lag_embed(scaler.transform(train_series), WINDOW)
    assert X.shape == (N_TRAIN, WINDOW)

    # 11,175 couplings from 80 rows: W must be regularised hard. On synthetic
    # i.i.d. returns, l2=1e-3 gave 37% coverage of the 90% interval and a
    # worse MAE than the zero forecast; l2=1 gave 88% and matched it.
    # Choose L2 on a validation window before trusting any forecast.
    model = Phi4Model(n_stocks=WINDOW, mu_global=True, lam_global=True, seed=0)
    model.fit(X, method="pl", l2=L2)

    rows = []
    for t in range(split, len(r)):
        d = model.forecast_distribution(r[t - (WINDOW - 1):t], scaler=scaler)
        lo, hi = d.interval(0.9)
        rows.append((r[t], d.mean(), lo, hi))
    y, mean, lo, hi = map(np.array, zip(*rows))
    print(f"MAE phi4  = {np.abs(mean - y).mean():.4%}")
    print(f"MAE zero  = {np.abs(y).mean():.4%}")
    print(f"direction hit rate  = {np.mean(np.sign(mean) == np.sign(y)):.0%}")
    print(f"90% interval coverage = {np.mean((y >= lo) & (y <= hi)):.0%} "
          f"(mean width {np.mean(hi - lo):.2%})")


if __name__ == "__main__":
    main()
