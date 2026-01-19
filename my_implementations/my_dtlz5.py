import numpy as np
from MoeaBench import mb


# 1. Component: Custom Problem Logic
# Inheriting from BaseMop ensures integration with the framework.
class MyProblem(mb.mops.BaseMop):
    """
    A custom bi-objective problem (ZDT1-like) implemented with NumPy.
    """
    def __init__(self):
        # M=2 Objectives, N=10 Variables, Bounds=[0, 1]
        super().__init__(M=2, N=10, xl=0.0, xu=1.0)

    def evaluation(self, X, n_ieq_constr=0):
        # Evaluation MUST be vectorized (X is a population matrix)
        # Objective 1: Minimize the value of the first variable
        f1 = X[:, 0]
        
        # Objective 2: Transformation designed for a convex front
        g = 1 + 9 * np.sum(X[:, 1:], axis=1) / (self.N - 1)
        f2 = g * (1 - np.sqrt(abs(f1 / g)))
        
        # Return objectives in dictionary 'F' (pop_size x M_objs)
        return {'F': np.column_stack([f1, f2])}



