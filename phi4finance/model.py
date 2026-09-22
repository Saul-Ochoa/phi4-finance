
import numpy as np
from tqdm import trange
from .sampler import MetropolisSampler

class Phi4Model:
    def __init__(self, n_stocks, lr=5e-3, mu_global=False, lam_global=True, proposal_range=3.0, seed=0):
        np.random.seed(seed)
        self.V=n_stocks
        self.lr=lr
        self.mu_global=mu_global
        self.lam_global=lam_global
        self.proposal_range=proposal_range
        self.W=np.random.randn(n_stocks,n_stocks)*0.01
        np.fill_diagonal(self.W,0)
        self.W=(self.W+self.W.T)/2
        self.mu=np.ones(n_stocks)*0.5
        self.lam=np.ones(n_stocks)*0.5
        self.a=np.zeros(n_stocks)

    def _sym(self):
        np.fill_diagonal(self.W,0)
        self.W=(self.W+self.W.T)/2

    def fit(self, data, epochs=300, mcmc_steps=1000, batch_size=64, verbose=True):
        N=data.shape[0]
        pbar=trange(epochs) if verbose else range(epochs)
        for epoch in pbar:
            idx=np.random.choice(N,batch_size,replace=False)
            batch=data[idx]
            exp_q_phi=batch.mean(axis=0)
            exp_q_phi2=(batch**2).mean(axis=0)
            exp_q_phi4=(batch**4).mean(axis=0)
            exp_q_phiphi=np.einsum("bi,bj->ij",batch,batch)/batch_size

            sampler=MetropolisSampler(self.W,self.mu,self.lam,self.a,proposal_range=self.proposal_range)
            samples_p=sampler.sample(n_steps=mcmc_steps,burn=mcmc_steps//3)
            exp_p_phi=samples_p.mean(axis=0)
            exp_p_phi2=(samples_p**2).mean(axis=0)
            exp_p_phi4=(samples_p**4).mean(axis=0)
            exp_p_phiphi=np.einsum("si,sj->ij",samples_p,samples_p)/len(samples_p)

            grad_W=-exp_p_phiphi+exp_q_phiphi
            grad_a=-exp_p_phi+exp_q_phi
            grad_mu=exp_p_phi2-exp_q_phi2
            grad_lam=exp_p_phi4-exp_q_phi4

            self.W-=self.lr*grad_W
            self._sym()
            if self.mu_global:
                self.mu-=self.lr*grad_mu.mean()
            else:
                self.mu-=self.lr*grad_mu
            if self.lam_global:
                self.lam-=self.lr*grad_lam.mean()
            else:
                self.lam-=self.lr*grad_lam
            self.a-=self.lr*grad_a

            if verbose and epoch%50==0:
                pbar.set_postfix({"m_q":f"{exp_q_phi.mean():.3f}","m_p":f"{exp_p_phi.mean():.3f}","Wm":f"{self.W.mean():.4f}"})
        return self

    def predict_conditional(self, known_dict, target_idx, n_samples=2000, mcmc_steps=5000):
        sampler=MetropolisSampler(self.W,self.mu,self.lam,self.a,proposal_range=self.proposal_range)
        fixed_idx=list(known_dict.keys())
        fixed_vals=list(known_dict.values())
        samples=sampler.sample_conditional(fixed_idx,fixed_vals,target_idx,n_steps=mcmc_steps)
        return samples

    def forecast_next_day(self, history_150, n_samples=3000):
        if self.V != len(history_150)+1:
            raise ValueError(f"Need V={len(history_150)+1}, got V={self.V}")
        known={i:history_150[i] for i in range(len(history_150))}
        return self.predict_conditional(known, target_idx=len(history_150), n_samples=n_samples)
