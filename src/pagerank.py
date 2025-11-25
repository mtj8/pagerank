import numpy as np
import math

DAMPING_FACTOR = 0.85
MAX_ITERS = 100
TOLERANCE = 1.0e-9





def page_rank(A: np.array, eps: float, max_iters: int) -> np.array:
    """
    Compute the PageRank vector R.

    args:
        A: normalized adjacency matrix (n x n)
        eps: convergence tolerance
        max_iters: maximum number of iterations
    
    returns:
        R: PageRank vector (n x 1)
    """
    n = A.shape[0]

    E = np.array([[1/n] for _ in range(n)])  # uniform "random surfer" vector

    R = E.copy()  # initial PageRank vector

    for iter in range(max_iters):
        R_new = (DAMPING_FACTOR * A @ R) + (1 - DAMPING_FACTOR) * E

        # convergence
        delta = np.linalg.norm(R_new - R, 1)
        if delta < eps:
            break

        R = R_new

    return R



def main():
    test1 = np.array([[0, 1/2, 1/2, 0], # A1
                      [1/3, 0, 0, 1/3],
                      [1/3, 1/2, 0, 1/3],
                      [1/3, 0, 1/2, 1/3]])
    A1 = test1
    pr1 = page_rank(A1, TOLERANCE, MAX_ITERS)
    print("Test 1 PageRank:\n", pr1)
    print("Sanity check: sum =", np.sum(pr1))
    
if __name__ == "__main__":
    main()

    