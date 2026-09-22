
import numpy as np
from .model import Phi4Model

def scaling_analysis(full_returns, volumes=[16,32,48,64], n_iter=10, epochs=200):
    results=[]
    V_full=full_returns.shape[1]
    base=Phi4Model(n_stocks=V_full, mu_global=True, lam_global=True)
    base.fit(full_returns, epochs=epochs, mcmc_steps=800, verbose=True)
    mu_g=base.mu.mean()
    lam_g=base.lam.mean()
    for V in volumes:
        w_means=[]; a_means=[]
        for it in range(n_iter):
            idx=np.random.choice(V_full,V,replace=False)
            sub=full_returns[:,idx]
            m=Phi4Model(n_stocks=V, mu_global=True, lam_global=True)
            m.mu[:]=mu_g; m.lam[:]=lam_g
            m.fit(sub, epochs=epochs, mcmc_steps=600, verbose=False)
            w_means.append(np.abs(m.W).mean())
            a_means.append(np.abs(m.a).mean())
        results.append((V, np.mean(w_means), np.mean(a_means)))
        print(f"V={V} <|w|>={np.mean(w_means):.5f} <|a|>={np.mean(a_means):.5f}")
    return results
