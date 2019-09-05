import torch
import numpy as np
import math
from scipy.special import binom
import pdb


def sigma_decomposition(s):
    lam = np.linalg.eig(s)[0]
    L = np.linalg.eig(s)[1] @ np.diag(np.sqrt(lam - np.min(lam)))
    return L, np.min(lam) 

def calc_error_bounds(Sigma, sigma, beta):

    p = beta.size
    k = np.count_nonzero(beta)
    L, sigma_mu = sigma_decomposition(Sigma)

    S = (beta != 0).astype(bool)
    nonzero_indices = np.nonzero(beta)[0]

    # Defer error probability until we know how costly E0 calculation is 
    # PE = np.zeros(k)
    # mutual inforinputions
    mutinf = np.zeros(k)

    # For each i, divide the support into disjoint subsets
    for i in range(1, k):

        # Choose a random subset
        s1 = np.random.choice(nonzero_indices, size=i, replace=False)
        s2 = np.array(list(set(nonzero_indices).difference(set(s1))))

        assert(s1.size + s2.size == k)

        S1 = np.zeros(p)
        S1[s1] = 1
        S1 = S1.astype(bool)

        S2 = np.zeros(p)
        S2[s2] = 1
        S2 = S2.astype(bool)

        delta, E0 = calc_errorexpon(k, S1, S2, beta, L, sigma, sigma_mu)
        mutinf[i] = calc_mutinf(delta, E0) 

    # Need to maximize over i
    log_factors = np.array([np.log(binom(p - k + i, i)) for i in range(k)])

    N = np.max(np.divide(log_factors, mutinf))

    return N

# Use pytorch's autograd features to calculate the necessary derivatives for our info theory calculations

'''
k : int, number of non-zero features
S1: i dimensional subset of the support (p dimensional boolean vector with i Trues)
S2: k - i dimensional complement (p dimensional boolean vector with k - i Trues) 
beta: Coefficient vector
L: latent factor loading inputrix in the factor model representation of the covariance inputrix
sigma: noise variance in linear model
sigma_mu: diagonal variance of design inputrix in factor model'''
def calc_errorexpon(k, S1, S2, beta, L, sigma, sigma_mu):

    p = beta.size
    i = np.count_nonzero(1* S1)

    # Convert inputs to torch tensors
    beta1 = torch.tensor(beta[S1], dtype=torch.float)
    beta2 = torch.tensor(beta[S2], dtype=torch.float)

    # Subspaces of L:
    Linv = np.linalg.pinv(L)
    Linv11 = torch.tensor(Linv[np.ix_(S1, S1)], dtype=torch.float)
    Linv12 = torch.tensor(Linv[np.ix_(S1, S2)], dtype=torch.float)
    Linv21 = torch.tensor(Linv[np.ix_(S2, S1)], dtype=torch.float)
    Linv22 = torch.tensor(Linv[np.ix_(S2, S2)], dtype=torch.float)

    Linv = torch.tensor(Linv, dtype=torch.float)
    L = torch.tensor(L, dtype=torch.float)

    delta = torch.tensor(0, dtype=torch.float, requires_grad=True)

    # Initialize the constants we use in our calculation
    A = torch.tensor(sigma_mu**2, dtype=torch.float)
    B = sigma**2 * (1 + delta)

    D = torch.eye(i) + A/B * torch.einsum('i,j->ij', beta1, beta1)
    # This is "\tilde{E}" in our notation, but why bother with the tildes here. Note that this form 
    # results from just multiplying through B by 1/(1 + delta)
    E = torch.squeeze(sigma**2/(1 - A/B * torch.chain_matmul(
                                          beta1[None, :], 
                                          torch.inverse(D), 
                                          beta1[:, None])))

    G = 1/A * torch.eye(k - i) + torch.pow(E, -1) * torch.einsum('i,j->ij', beta2, beta2)

    # intermediate alpha tilde constants
    alpha0 = torch.squeeze(2 * (torch.pow(E, -1) + torch.pow(E, -2) * torch.chain_matmul(
                                                                      beta2[None, :], 
                                                                      torch.inverse(G), 
                                                                      beta2[:, None])))

    alpha11 = 1/sigma**4 * torch.chain_matmul(
                           torch.inverse(D),
                           torch.einsum('i,j->ij', beta1, beta1),
                           torch.inverse(D))

    alpha22 = torch.pow(A, -2) * torch.pow(E, -2) * torch.chain_matmul(
                                                    torch.inverse(G),
                                                    torch.einsum('i,j->ij', beta2, beta2),
                                                    torch.inverse(G))

    alpha33 = 1/sigma**4 * torch.pow(E, -2) * torch.chain_matmul(
                                              torch.inverse(D),
                                              torch.einsum('i,j->ij', beta1, beta2),
                                              torch.inverse(G),
                                              torch.einsum('i,j->ij', beta2, beta2),
                                              torch.inverse(G),
                                              torch.einsum('i,j->ij', beta2, beta1),
                                              torch.inverse(D))

    alpha12 = torch.pow(A, -1) * torch.pow(E, -2)/sigma**2 * torch.chain_matmul(
                                                       torch.inverse(D),
                                                       torch.einsum('i,j->ij', beta1, beta2),
                                                       torch.inverse(G))

    alpha13 = torch.pow(E, -1)/sigma**4 * torch.chain_matmul(
                                       torch.inverse(D),
                                       torch.einsum('i,j->ij', beta1, beta2), 
                                       torch.inverse(G),
                                       torch.einsum('i,j->ij', beta2, beta1), 
                                       torch.inverse(D))

    alpha23 = torch.pow(A, -1) * torch.pow(E, -2)/sigma**2 * torch.chain_matmul(
                                                       torch.inverse(G),
                                                       torch.einsum('i,j->ij', beta2, beta2),
                                                       torch.inverse(G), 
                                                       torch.einsum('i,j->ij', beta2, beta1), 
                                                       torch.inverse(D))

    # Gamma inputrix
    G11 = torch.pow(A/(1 + delta), -1) * (torch.eye(i) - torch.inverse(D)) \
          - 1/sigma**4 * torch.chain_matmul(
                         torch.inverse(D), 
                         torch.einsum('i,j->ij', beta1, beta2),
                         torch.inverse(G),
                         torch.einsum('i,j->ij', beta2, beta1),
                         torch.inverse(D)) \
          - torch.pow(alpha0, -1) * (alpha11 + alpha33 + alpha13) - Linv11

    G12 = -1 * (torch.pow(A, -1)/sigma**2 * torch.chain_matmul(
                                      torch.inverse(D),
                                      torch.einsum('i,j->ij', beta1, beta2),
                                      torch.inverse(G)) \
          + 2 * torch.pow(alpha0, -1) * alpha12 -  Linv12)

    G21 = -1 * (torch.pow(A, -1)/sigma**2 * torch.chain_matmul(
                                      torch.inverse(G),
                                      torch.einsum('i,j->ij', beta2, beta1),
                                      torch.inverse(D)) \
          + 2 * torch.pow(alpha0, -1) * alpha23 -  Linv21)

    G22 = torch.pow(A, -1) * (torch.eye(k - i) - torch.pow(A, -1) * torch.inverse(G))\
          - torch.pow(alpha0, -1) * alpha22 + Linv22

    # Assemble the final error exponent #
    E0 = 1/2 * (torch.log(alpha0 * sigma**2) - 2 * torch.log(torch.det(Linv)) + torch.log(torch.det(G))\
                + (1 + delta) * torch.log(torch.det(D)) + (k - i) * torch.log(A)\
                - torch.log(torch.tensor(2, dtype=torch.float))\
                + torch.log(torch.det(G11)*torch.det(torch.chain_matmul(G21, torch.inverse(G11), G21.t())\
                                                    + torch.chain_matmul(G21, torch.inverse(G11), G12)\
                                                    + torch.chain_matmul(G12.t(), torch.inverse(G11), G21.t())\
                                                    + torch.chain_matmul(G12.t(), torch.inverse(G11), G12)\
                                                    + G22)))

    return delta, E0


def calc_mutinf(delta, E0):

    # Take the gradient with respect to delta
    E0.backward()
    mutinf = delta.grad
    mutinf = mutinf.detach().cpu().numpy().astype(float)
    return mutinf
