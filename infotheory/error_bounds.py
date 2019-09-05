import torch
import numpy as np
import math

# Use pytorch's autograd features to calculate the necessary derivatives for our info theory calculations

'''
k : int, number of non-zero features
S1: i dimensional subset of the support (p dimensional boolean vector with i Trues)
S2: k - i dimensional complement (p dimensional boolean vector with k - i Trues) 
beta: Coefficient vector
L: latent factor loading matrix in the factor model representation of the covariance matrix
sigma: noise variance in linear model
sigma_mu: diagonal variance of design matrix in factor model'''
def calc_gradient(k, S1, S2, beta, L, *constants):

    p = beta.size
    i = np.count_nonzero(1* S1)

    # Convert inputs to torch tensors
    beta1 = torch.tensor(beta[S1])
    beta2 = torch.tensor(beta[S2])

    # Subspaces of L:
    Linv = np.inverse(L)
    Linv11 = torch.tensor(Linv[np.ix_(S1, S1)])
    Linv12 = torch.tensor(Linv[np.ix_(S1, S2)])
    Linv21 = torch.tensor(Linv[np.ix_(S2, S1)])
    Linv22 = torch.tensor(Linv[np.ix_(S2, S2)])

    Linv = torch.tensor(Linv)
    L = torch.tensor(L)

    delta = torch.tensor(0, dtype=torch.float32, requires_grad=True)

    # Initialize the constants we use in our calculation
    A = sigma_mu**2
    B = sigma**2 * (1 + delta)
    D = torch.eye(i) + A/B * torch.addr(mat=0, vec1=beta1, vec2=beta1)
    # This is "\tilde{E}" in our notation, but why bother with the tildes here. Note that this form 
    # results from just multiplying through B by 1/(1 + delta)
    E = sigma**2/(1 - A/B * torch.chain_matmul(beta1.t, torch.inverse(D), beta1))
    G = 1/A * np.eye(k - i) + torch.pow(E, -1) * torch.addr(mat=0, vec1=beta2, vec2=beta2)

    # intermediate alpha tilde constants
    alpha0 = 2 * (torch.pow(E, -1) + torch.pow(E, -2) * torch.chain_matmul(beta2.t, torch.inverse(G), beta2))
    alpha11 = 1/sigma**4 * torch.addr(mat=0, vec1=torch.mm(torch.inverse(D), beta1), vec2=torch.mm(beta1.t, torch.inverse(D)))
    alpha22 = torch.pow(A, -2) * torch.pow(E, -2) * \
              torch.addr(mat=0, vec1=torch.mm(torch.inverse(G), beta2), vec2=torch.mm(beta2.t, torch.inverse(G)))  

    alpha33 = 1/sigma**4 * torch.pow(E, -2) * torch.chain_matmul(
                                              torch.inverse(D),
                                              torch.addr(mat=0, vec1=beta1, vec2=beta2),
                                              torch.inverse(G),
                                              torch.addr(mat=0, vec1=beta2, vec2=beta2),
                                              torch.inverse(G),
                                              torch.addr(mat=0, vec1=beta2, vec2=beta1),
                                              torch.inverse(D))
    alpha12 = np.pow(A, -1) * np.pow(E, -2)/sigma**2 * torch.chain_matmul(
                                                       torch.inverse(D),
                                                       torch.addr(mat=0, vec1=beta1, vec2=beta2),
                                                       torch.inverse(G))
    alpha13 = np.pow(E, -1)/sigma**4 * torch.chain_matmul(
                                       torch.inverse(D),
                                       torch.addr(mat=0, vec1=beta1, vec2=beta2), 
                                       torch.inverse(G),
                                       torch.addr(mat=0, vec1=beta2, vec2=beta1), 
                                       torch.inverse(D))
    alpha23 = np.pow(A, -1) * np.pow(E, -2)/sigma**2 * torch.chain_matmul(
                                                       torch.inverse(G),
                                                       torch.addr(mat=0, vec1=beta2, vec2=beta2),
                                                       torch.inverse(G), 
                                                       torch.addr(mat=0, vec1=beta2, vec2=beta1), 
                                                       torch.inverse(D))
    # Gamma matrix
    G11 = torch.pow(A/(1 + delta), -1) * (torch.eye(i) - torch.inverse(D)) \
          + 1/sigma**4 * torch.chain_matmul(
                         torch.inverse(D), 
                         torch.addr(mat=0, vec1=beta1, vec2=beta2),
                         torch.inverse(G),
                         torch.addr(mat=0, vec1=beta2, vec2=beta1),
                         torch.inverse(D)) \
          - torch.pow(alpha0, -1) * (alpha11 + alpha33 + alpha13) + Linv11

    G12 = -1 * (torch.pow(A, -1)/sigma**2 * torch.chain_matmul(
                                      torch.inverse(D),
                                      torch.addr(mat = 0, vec1 = beta1, vec2=beta2),
                                      torch.inverse(G)) \
          + 2 * torch.pow(alpha0, -1) * alpha12 -  Linv12)

    G21 = -1 * (torch.pow(A, -1)/sigma**2 * torch.chain_matmul(
                                      torch.inverse(D),
                                      torch.addr(mat = 0, vec1 = beta1, vec2=beta2),
                                      torch.inverse(G)) \
          + 2 * torch.pow(alpha0, -1) * alpha23 -  Linv21)

    G22 = torch.eye(k - i) + torch.inverse(G) - torch.pow(alpha0, -1) * alpha22 + Linv22

    # Assemble the final error exponent. Not sure if we should include n??

    E0 = 1/2 * (torch.log(2 * math.pi * sigma**2) + torch.log(torch.det(L)) - torch.log(torch.det(G))\
                + (1 + delta) * torch.log(torch.det(D)) + (k - i) * torch.log(2 * math.pi * A) - torch.log(4 * math.pi)\
                + torch.log(alpha0) \
                + torch.log(torch.det(torch.mm(G11, torch.chain_matmul(G21, torch.inverse(G11), G21.t)\
                                                    + torch.chain_matmul(G21, torch.inverse(G11), G12)\
                                                    + torch.chain_matmul(G12.t, torch.inverse(G11), G21.t)\
                                                    + torch.chain_matmul(G12.t, torch.inverse(G11), G12)\
                                                    + G22))))

    # Take the gradient with respect to delta
    E0.backward()
    mutinf = E0.grad
