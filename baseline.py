from analysisTool import pipeline
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys


def randomA(D=250, F=250):

    M = np.random.randn(D, F)
    randomStiefel, _ = np.linalg.qr(
        M
    )  # Apply Gram-Schmidt algorithm to the columns of M

    return randomStiefel


def fitBeta(A, X_train_reshape: pd.DataFrame, Y_train: pd.DataFrame):

    predictors = (
        X_train_reshape @ A
    )  # the dataframe of the 10 factors created from A with the (date, stock) in index
    targets = Y_train.T.stack()
    beta = np.linalg.inv(predictors.T @ predictors) @ predictors.T @ targets

    return beta.to_numpy()


def checkOrthonormality(A):

    bool = True
    D, F = A.shape
    Error = pd.DataFrame(A.T @ A - np.eye(F)).abs()

    if any(Error.unstack() > 1e-6):
        bool = False

    return bool


def metric_train(A, beta, X_train_reshape: pd.DataFrame, Y_train: pd.DataFrame):

    if not checkOrthonormality(A):
        print("fail")
        return -1.0

    Ypred = (X_train_reshape @ A @ beta).unstack().T
    Ytrue = Y_train

    Ytrue = Ytrue.div(np.sqrt((Ytrue**2).sum()), 1)
    Ypred = Ypred.div(np.sqrt((Ypred**2).sum()), 1)

    meanOverlap = (Ytrue * Ypred).sum().mean()

    return meanOverlap


def cfs_search(
    X_train_reshape: pd.DataFrame, Y_train: pd.DataFrame, y_stack,  size_search: int = 50
):
    A_full = randomA(F=size_search)
    # randomNormedA(D=250, F = 50)
    # define variables for cfs algorithm
    

    X_stack = [
        (X_train_reshape @ A_full)
        .loc[(slice(None), k), :]
        .reset_index(level=1)
        .drop(["stocksID"], axis=1)
        for k in range(50)
    ]

    # perform cfs search
    result = []
    for X, y in zip(X_stack, y_stack):
        selected_features, _ = pipeline(X, y, min_features=10)
        result.append(selected_features)

    # keep best 10 features
    flatlist = [element for sublist in result for element in sublist]
    d_ = dict()
    for s in flatlist:
        d_[s] = d_.get(s, 0) + 1
    sorted_keys = sorted(d_, key=d_.get, reverse=True)
    columns = sorted_keys[:10]

    # find matrix A
    A = A_full[:, columns]

    # fit beta
    beta = fitBeta(A, X_train_reshape, Y_train)

    # compute the metric on the training set and keep the best result

    m = metric_train(A, beta, X_train_reshape, Y_train)
    return m, A, beta


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

    return X_train, Y_train, X_train_reshape


def main():
    args = sys.argv[1:]
    X_train, Y_train, X_train_reshape = retrieve_data()


    Niter = 100000
    maxMetric = -1

    #np.random.seed(1234)
    pbar = tqdm(range(Niter))
    for iteration in pbar:
        
        # Generate a uniform random Stiefel matric A and fit beta with minimal mean square prediction error on the training data set
        
        A = randomA(F=30)
        
        beta = fitBeta(A, X_train_reshape, Y_train)
        
        # compute the metric on the training set and keep the best result   
        
        m = metric_train(A, beta, X_train_reshape, Y_train)
        pbar.set_postfix({'metric_train': m, 'max': maxMetric})
        if m > maxMetric:
            maxMetric = m
            A_QRT = A
            np.save("A.baseline", A_QRT)
            beta_QRT = beta  

if __name__ == "__main__":
    main()