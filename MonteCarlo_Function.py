#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

from numpy.random import default_rng
import math
from math import comb, factorial
from typing import Optional


# In[2]:


# Monte Carlo integral over (0,1)^k:  ∫_{[0,1]^k} g(x) dx = E[g(U)],  U ~ Unif((0,1)^k)
def mc_integrate_unit_cube(g_vec, k, n_samples = 1_000_000, batch = 100_000, seed = None):
    """
    g_vec: callable(U) -> shape (m,) values for U shape (m,k)
    k: dimension
    returns (estimate, standard_error)
    """
    rng = np.random.default_rng(seed)
    total = 0.0
    total2 = 0.0
    done = 0
    
    while done < n_samples:
        m = min(batch, n_samples - done)
        U = rng.random((m, k))                # samples in [0,1)
        vals = g_vec(U).astype(float)         # vectorized g
        total += vals.sum()
        total2 += np.square(vals).sum()
        done += m
    mean = total / n_samples
    var = total2 / n_samples - mean**2
    se = np.sqrt(max(var, 0.0) / n_samples)
    
    return mean, se


# In[3]:


# function g(u):
def g_vec(u, N, k):
    u = u[:, :k-1]
    s  = np.sum(u, axis=1)
    mx = np.max(u, axis=1)
    mask = (s < 1.0)                           # indicator 1_{sum(u)<1}
    log_g = (
        N * np.log1p(-mx)                      # N*log(1 - max u_i)
        - (N - k) * np.log(k - s)              # -(N-k)*log(k - sum u_i)
        - 2.0 * np.sum(np.log1p(-u), axis=1)   # -2*sum log(1 - u_i)
    )
    out = np.exp(log_g)*mask
    return out


# In[4]:


def probs_all(N):
    probs = []
    
    for k in range (2, N+1):
        C = math.comb(N, k) * k * ((k-1) ** (N - k)) / N
    
        I_hat, stderr_I = mc_integrate_unit_cube(lambda U: g_vec(U, N, k), k, n_samples=2_000_000, batch=100_000, seed=0)
    
        p = C * I_hat
        se = C * stderr_I
        probs.append(p)
    
    return probs

