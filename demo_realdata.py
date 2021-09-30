"""
This code provides a simple realization when no local data is available
It contains:
    1. fetch data, read information and make partition
    2. learn the hyperparameters from the bilevel model
    3. test the performance
"""

#%%
import numpy as np
import time
import cvxpy as cp
from sklearn.datasets import fetch_openml
import scipy.sparse as sp

from ip_dca import bilevel_SVM_iP_DCA, test

#%%
## data partition
def data_partition(X, y, T):
    N_all = X.shape[0]
    if N_all != y.shape[0]:
        print('error: X and y are inconsistent')
        return
    if N_all%T != 0:
        print('warning：some data will be thrown to make a equal partition')
        N_all = N_all - N_all%T
    N_1 = N_all // T  # scale of each validation data set
    J_val, J_trn = [], []
    for i in range(T):
        tmp = np.arange(i*N_1, (i+1)*N_1)
        J_val.append(tmp)
        J_trn.append(np.setdiff1d(np.arange(0, N_all), tmp))
    return X[0:N_all], y[0:N_all], J_val, J_trn

#%%
# load data
whole_data = fetch_openml(name='australian') # you could change dataset here 
X = sp.csr_matrix(whole_data.data.values.astype(float))
y = whole_data.target.values.astype(float)

# check if the problem is binary problem and turn the label to {-1, 1} problem
cs = np.unique(y)
assert (len(cs) == 2), 'this code is only for binary svm problem, however, the data is not binary'

y, cs = y+100, cs+100
y[y==cs[0]] = -1
y[y==cs[1]] = 1

#%%
# read info and do data partition
T, [M, n] = 3, X.shape
nVal = M//(2*T)
nTrn = (T - 1) * nVal

shuffle = 1
ordering = np.arange(0, M)
if shuffle:
    np.random.shuffle(ordering)
    X, y = X[ordering], y[ordering]

[XTrn, yTrn, J_val, J_trn] = data_partition(X[ordering[0:(T*nVal)], :], y[ordering[0:(T*nVal)]], T)
[XTest, yTest] = X[ordering[(T*nVal):], :], y[ordering[(T*nVal):]]

#%%
# Learn hyperparameters in bilevel model
## set parameters and initial guess
opt = dict(
    itr = 500, epsilon = 1e-3, beta = 1., delta = 5., rho = 1e-2, tol = 1e-3, 
    printflag = 1, svm_solver = cp.ECOS, main_solver = cp.ECOS
    )
lambda_bnd, w_bar_bnd = [1e-4, 1e4], [1e-6, 1.5]
initial_guess = dict(mu = 1., w_bar = 1e-6*np.ones(n), W = np.zeros([T, n]), c = np.zeros(T))

tic = time.time()
result_learn = bilevel_SVM_iP_DCA(XTrn, yTrn, J_val, J_trn, lambda_bnd, w_bar_bnd, initial_guess, opt)
time_cost = time.time() - tic 
print('time cost: {:.2f}s'.format(time_cost))
lam, w_bar = result_learn['var']['lam'], result_learn['var']['w_bar']
with np.printoptions(precision=3, suppress=True): print('  lam =  {:.3f}'.format(lam), '\nw_bar = ', w_bar)

#%%
# Test the performance of the learned hyperparamters
result_test = test(XTrn, yTrn, J_val, J_trn, XTest, yTest, lam, w_bar)
train_err, cross_err_1, test_err = result_test['train_err'], result_test['cross_err'], result_test['test_err']
cross_err_2 = float(result_learn['obj']['F'])
print( "{:.5f}, {:.5f}, {:.5f}, {:.5f}".format(train_err, cross_err_1, cross_err_2, test_err))
