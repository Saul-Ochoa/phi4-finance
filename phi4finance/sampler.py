
import numpy as np

class MetropolisSampler:
    def __init__(self, W, mu, lam, a, proposal_range=3.0):
        self.W = W
        self.mu = mu
        self.lam = lam
        self.a = a
        self.proposal_range = proposal_range
        self.V = len(a)

    def action(self, phi):
        term_w = - np.einsum("i,ij,j->", phi, self.W, phi)
        term_mu = np.sum(self.mu * phi**2)
        term_lam = np.sum(self.lam * phi**4)
        term_a = - np.dot(self.a, phi)
        return term_w + term_mu + term_lam + term_a

    def sample(self, n_steps=2000, burn=500, init=None):
        V = self.V
        phi = np.random.uniform(-1,1,size=V) if init is None else init.copy()
        samples = []
        for step in range(n_steps):
            for i in range(V):
                phi_prop = phi.copy()
                phi_prop[i] = np.random.uniform(-self.proposal_range/2, self.proposal_range/2)
                dS = self.action(phi_prop) - self.action(phi)
                if dS < 0 or np.random.rand() < np.exp(-dS):
                    phi = phi_prop
            if step >= burn:
                samples.append(phi.copy())
        return np.array(samples)

    def sample_conditional(self, fixed_idx, fixed_values, target_idx, n_steps=5000):
        V = self.V
        phi = np.random.uniform(-1,1,size=V)
        for k,v in zip(fixed_idx, fixed_values):
            phi[k]=v
        out=[]
        for step in range(n_steps):
            targets = [target_idx] if isinstance(target_idx,int) else list(target_idx)
            for i in targets:
                if i in fixed_idx:
                    continue
                phi_prop = phi.copy()
                phi_prop[i]=np.random.uniform(-1.5,1.5)
                dS = self.action(phi_prop)-self.action(phi)
                if dS < 0 or np.random.rand() < np.exp(-dS):
                    phi=phi_prop
            if step>1000:
                out.append(phi[target_idx] if isinstance(target_idx,int) else phi[targets].copy())
        return np.array(out)
