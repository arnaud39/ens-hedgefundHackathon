import numpy as np
import pandas as pd
from pymoo.problems.functional import FunctionalProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

import numba
from numba.experimental import jitclass


@jitclass(
    [
        (
            "X_train_reshape",
            numba.types.Array(dtype=numba.types.float32, ndim=2, layout="C"),
        ),
        ("Y_train", numba.types.Array(dtype=numba.types.float32, ndim=2, layout="C")),
    ]
)
class Genetic_obj:
    def __init__(self, X_train_reshape: np.array, Y_train: np.array):
        self.X_train_reshape = X_train_reshape
        self.Y_train = Y_train

    def fitBeta(self, A: np.array):
        predictors_ = self.X_train_reshape @ A
        targets_ = self.Y_train.T.copy().reshape((-1))
        beta = np.linalg.inv(predictors_.T @ predictors_) @ predictors_.T @ targets_
        return beta

    def metric_train(self, A: np.array, beta: np.array):
        Ypred_ = (self.X_train_reshape @ A @ beta).reshape((-1, 50)).T
        Ytrue_ = self.Y_train

        Ytrue_ = Ytrue_ / np.sqrt((Ytrue_**2).sum(axis=0))
        Ypred_ = Ypred_ / np.sqrt((Ypred_**2).sum(axis=0))
        meanOverlap = (Ytrue_ * Ypred_).sum(axis=0).mean()

        return meanOverlap

    def objectiv_function(self, x: np.array):
        A_raw = x.reshape((250, 10))
        A, _ = np.linalg.qr(A_raw)

        # fit beta
        beta = self.fitBeta(A)

        # compute the metric on the training set and keep the best result

        m = self.metric_train(A, beta)
        return m
