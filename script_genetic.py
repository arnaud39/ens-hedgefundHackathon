import numpy as np
import pandas as pd
from pymoo.problems.functional import FunctionalProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch


from pymoo.termination import get_termination
from pymoo.optimize import minimize

from genetic_research import Genetic_obj
import tensorflow as tf


def retrieve_data():
    path = ""

    X_train = pd.read_csv(path + "X_train.csv", index_col=0, sep=",")
    X_train.columns.name = "date"

    Y_train = pd.read_csv(path + "Y_train.csv", index_col=0, sep=",")
    Y_train.columns.name = "date"

    X_train_reshape = pd.concat(
        [X_train.T.shift(i + 1).stack(dropna=False) for i in range(250)], 1
    ).dropna()
    X_train_reshape.columns = pd.Index(range(1, 251), name="timeLag")

    return tf.convert_to_tensor(
        np.ascontiguousarray(X_train_reshape.to_numpy(), dtype=np.float32)
    ), tf.convert_to_tensor(np.ascontiguousarray(Y_train.to_numpy(), dtype=np.float32))


def parametersTransform(A, beta, D=250, F=10):

    if A.shape != (D, F):
        print("A has not the good shape")
        return

    if beta.shape[0] != F:
        print("beta has not the good shape")
        return

    output = np.hstack((np.hstack([A.T, beta.reshape((F, 1))])).T)

    return output


def main():

    import time

    # x0 = np.load("x0.npy")
    instance = Genetic_obj(*retrieve_data())
    objs = [
        lambda x: -1
        * instance.objectiv_function(
            tf.convert_to_tensor(np.ascontiguousarray(x, dtype=np.float32))
        ),
    ]

    n_var = 250 * 10

    problem = FunctionalProblem(
        n_var,
        objs,
        xl=np.ones(2500) * -3,
        xu=np.ones(2500) * 3,
    )

    algorithm = NSGA2(pop_size=300)  # PatternSearch(x0=x0)

    res = minimize(problem, algorithm, ("n_gen", 200), seed=1, verbose=True)
    print(f"Best solution found: \nX = {res.X}\nF = {res.F}\nCV= {res.CV}")

    np.save(f"x_genetic_{res.F[0]}", res.X)

    A_raw = tf.convert_to_tensor(res.X, dtype=np.float32).reshape([250, 10])
    A, _ = tf.linalg.qr(A_raw)
    beta = instance.fitBeta(A)
    # from output to csv file...
    output = parametersTransform(A, beta)
    pd.DataFrame(output).to_csv(f"ArnaudPetitsubmission_{res.F[0]}.csv")


if __name__ == "__main__":
    main()
