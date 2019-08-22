import numpy as np
from scipy.linalg import eigh
import pdb

# Local linear approximation algorithm for SCAD/MCP using FISTA to do optimization
# Implemented from https://orfe.princeton.edu/~jqfan/papers/14/StrongOracle14.pdf

# Expertiments to do: Test against Pycasso
# What about initializing with the Pycasso estimate? What happens then?

# Custom implementation of Fast iterative thresholding algorithm
def FISTA(x0, A, b, alpha, max_iter = 1000, tol=1e-6):

    # Minimize ||A x - b||^2 + \alpha |x|_1, where alpha can be a vector of diff.
    # values

    # Thresholding operator: alpha is a vector of the same size as x that allows
    # one to separately set the thresholding for each entry of x
    T_lambda = lambda alpha, x: np.multiply(np.sign(x), np.maximum(np.abs(x) - alpha, 0))
    # Iterative shrinkage operator for L1 regularization
    p_L = lambda y, A, b, alpha, t: T_lambda(alpha * t, y - 2 * t * A.T @ (A @ y - b))    

    # Used to assess stopping criteria
    loss_fn = lambda y, A, b, alpha: np.linalg.norm(A @ y - b)**2 + alpha @ np.abs(y)
    
    # Calculate the Lipshitz constant associated with this problem 
    # Calculate the largest eigenvalue of the Hessian
    L = 2 * eigh(A.T @ A, eigvals_only=True, eigvals=(A.shape[1] - 1, A.shape[1] - 1))

    # Main FISTA loop
    y = x0

    # Need to keep track of information adjacent to the iterations
    # [x_{k -1} x_k]
    x = [x0, 0]
    # [t_k t_{k + 1}]
    t = [1, 0]

    loss = np.zeros(max_iter)

    for k in range(max_iter):
        x[1] = p_L(y, A, b, alpha, 1/L)
        # FISTA should always give decreasing values
        loss[k] = loss_fn(x[1], A, b, alpha)
        try:
            if k >= 1:
                assert(loss[k] <= loss[k-1])
        except:
            continue

        # Stopping criteria: Difference in function values should be less than        
        if k > 1:
            if np.abs(loss[k] - loss[k-1]) < tol:
                break

        t[1] = (1 + np.sqrt(1 + 4 * t[0]**2))/2
        y = x[1] + (t[0] - 1)/t[1] * (x[1] - x[0])
        
        # Shift indices for the next loop
        t[0] = t[1]
        x[0] = x[1]

    # Final loss and tolerance
    final_loss = loss_fn(x[1], A, b, alpha)
    final_tol = np.abs(final_loss - loss_fn(x[0], A, b, alpha))

    return (x[1], final_loss, final_tol)


def LLA(beta_0, X, y, penalty_fn, penalty_fn_args=(), max_iter=100, tol=1e-6):
    ''' beta_0: Suitable initialization of the weights
        penalty_fn: Handle to the derivative of the penalty (i.e SCAD or MCP)
        penalty_fn_args: Tuple of arguments, beyond beta, to send to the loss 
        function (i.e. regularization strengths)
        max_iter: Maximum number of iterations to allow the LLA'''

    # Compute the initial adaptive weight vector
    w = penalty_fn(np.abs(beta_0), *penalty_fn_args)

    # Keep track of adjacent iterations
    beta = [beta_0, 0]

    for m in range(max_iter):

        # Use FISTA to solve the weighted L1 optimization
        beta[1] = FISTA(beta[0], X, y, w)[0]

        pdb.set_trace()

        # Stopping criteria:
        if np.linalg.norm(beta[1] - beta[0]) < tol:
            break

        # Update the weighting
        w = penalty_fn(np.abs(beta[1]), *penalty_fn_args)

        # Shift indices
        beta[0] = beta[1]

    return beta

# Derivative of SCAD penalty
def dSCAD(x, a, alpha):

    return np.array([np.maximum(a * alpha - np.abs(xi), 0)/(a -1)
                      if np.abs(xi) > alpha else alpha for xi in x])

# Derivative of the MCP penalty:
def dMCP(x, a, alpha):

    return np.maximum(alpha - np.abs(x)/a, 0)