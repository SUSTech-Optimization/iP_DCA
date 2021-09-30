'''
This code is to show the performance of iP-DCA on the real world problem
It requires the libsvm datasets in the folder ./datasets/
the corresponding datasets can be download from the following url:
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
'''

#%%
import numpy as np
import cvxpy as cp
import time
from sklearn.datasets import load_svmlight_file

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
print('large-scale test with opensource solver')
from itertools import product
# datasets = ['australian_scale', 'breast-cancer_scale', 'diabetes_scale']
datasets = ['mushrooms', 'phishing']
repeat_time = 20
# paras = dict(epsilon = [0., 1e-2, 1e-4], tol = [1e-2, 1e-3])
paras = dict(epsilon = [0., 1e-2, 1e-4], tol = [1e-2])
param_values = [v for v in paras.values()]
n_setting = len(list(product(*param_values)))
recording = []
for data in datasets:
    print('dataset: ', data)
    # load data
    [X, y] = load_svmlight_file("datasets/"+data)
    
    # check if the problem is binary problem and turn the label to {-1, 1} problem
    cs = np.unique(y)
    if len(cs) != 2:
        print('error: this is not a binary problem')
    else:
        y, cs = y+100, cs+100
        y[y==cs[0]] = -1
        y[y==cs[1]] = 1
        
    T, [M, n] = 3, X.shape
    nVal = M//(2*T)
    nTrn = (T - 1) * nVal
    
    opt = dict(
        itr = 500, epsilon = 1e-3, beta = 1., delta = 5., rho = 1e-2, tol = 1e-3, 
        printflag = 0, svm_solver = cp.ECOS, main_solver = cp.SCS
        )
    initial_guess = dict(
        mu = 1., w_bar = 1e-6*np.ones(n), W = np.zeros([T, n]), c = np.zeros(T)
        )
    lambda_bnd, w_bar_bnd = [1e-4, 1e4], [1e-6, 1.5]
    
    for k in range(repeat_time):
        print('repeat_time: ', k+1)
        # shuffle data
        ordering = np.arange(0, M)
        np.random.shuffle(ordering)
        X, y = X[ordering], y[ordering]

        [XTrn, yTrn, J_val, J_trn] = data_partition(X[ordering[0:(T*nVal)], :], y[ordering[0:(T*nVal)]], T)
        [XTest, yTest] = X[ordering[(T*nVal):], :], y[ordering[(T*nVal):]]
        
        counter = 0
        for opt['epsilon'], opt['tol'] in product(*param_values):
            counter += 1
            print( 'setting: {:<d} of {:<d}'.format(counter, n_setting), end = ' ')
            tic = time.time()
            result = bilevel_SVM_iP_DCA(XTrn, yTrn, J_val, J_trn, lambda_bnd, w_bar_bnd, initial_guess, opt)
            time_cost = time.time() - tic
            print( 'time cost: {:.2f}s'.format(time_cost) )
            mu, w_bar = result['var']['mu'], result['var']['w_bar']
            result_test = test(XTrn, yTrn, J_val, J_trn, XTest, yTest, 1/mu, w_bar)
            train_err, cross_err_1, test_err_1 = result_test['train_err'], result_test['cross_err'], result_test['test_err']
            cross_err_2 = float(result['obj']['F'])
            record = [data, k+1, opt['epsilon'], opt['tol'], cross_err_1, cross_err_2, test_err_1, time_cost]
            recording.append(record)
            
            file = open("results_1.5_large.txt", "a")
            file.write("%20s %2d %.3f %.3f %.7f %.7f %.7f %3.3f\n" % 
                       (record[0], record[1], record[2], record[3], record[4], record[5], record[6], record[7]))
            file.close()
    
#%%

collect = []
for data in datasets:
    for epsilon, tol in product(*param_values):
        tmp = []
        for record in recording:
            if (record[0] == data) & (record[2] == epsilon) & (record[3] == tol):
                tmp.append(record[4:])
        collect.append([data, epsilon, tol, tmp])

for item in collect:
    print("{:20s} {:.0e} {:.0e}".format(item[0], item[1], item[2]), end = '    ')
    tmp = np.array(item[3])
    mean = np.mean(tmp, 0)
    std = np.std(tmp, 0)
    print("{:.4f}({:.4f})  {:.4f}({:.4f})  {:3.1f}({:2.1f})".format(mean[1], std[1], mean[2], std[2], mean[3], std[3]) )
    
for item in collect:
    print("{:20s} {:.0e} {:.0e}".format(item[0], item[1], item[2]), end = '    ')
    tmp = np.array(item[3])
    mean = np.mean(tmp, 0)
    std = np.std(tmp, 0)
    print("{:.2f}({:.2f})  {:.2f}({:.2f})  {:3.1f}({:2.1f})".format(mean[1], std[1], mean[2], std[2], mean[3], std[3]) )