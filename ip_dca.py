#%%
import numpy as np
import cvxpy as cp

#%%
def bilevel_SVM_iP_DCA(X, y, J_val, J_trn, lambda_bnd, w_bar_bnd, initial_guess = {}, opt = {}):
    '''A Solver for K-fold SVM problem, aim to find the hyperparameters for smallest cross-validation
    specifically, the SVM model we considrated here is a classifier only for binaray problem
    
    Arguments
    ---------
    X, y:                   features and labels
    J_val, J_trn:           index sets for validation and train
    lambda_bnd, w_bar_bnd:  range of hyper-parameters
    initial_guess:          dict variable contains initial guess for mu/w_bar/W/c
    opt: dict variable contains contorllers of iP-DCA
        beta_k/epsilon/rho/delta:   parameters of algorithm
        itr:        maximum step of iterations
        tol:        torlerance used in stop test
        printflag:  0 for no output of run info
                    1 for output the process info
        svm_solver:  solver for svm subproblem
        main_solver: sovler for the main subproblem
    
    NOTE: 
        1. When solvers are not specified, we will try to use GUROBI or MOSEK if one of them is available;
           and the prioty of GUROBI is higher; otherwise we will use ECOS instand;
        2. For large-scale problem, we highly recommand to use cp.SCS for the main subproblem, i.e.,
           to specify main_solver = cp.SCS
           SCS is a first-order solver designed based on OpenMP
           and suppose to attain better performance than other solvers on large-scale problem.
        3. One could check the installed solver with print(cp.installed_solvers()) # import cvxpy as cp
    
    Return
    ---------
    result = dict(obj = obj, var = var)
    where
    obj = dict(F = F1.value)
    var = dict(mu = mu_k, w_bar = w_bar_k, W = W_k, c = c_k, lam = 1./mu_k)
    ---------
    '''
    # Extracting basic info
    n = X.shape[1]
    T, nVal, nTrn = len(J_val), len(J_val[0]), len(J_trn[0])
    lambda_lb, lambda_ub = lambda_bnd[0], lambda_bnd[1]
    w_bar_lb, w_bar_ub = w_bar_bnd[0], w_bar_bnd[1]
    
    # Reading parameters and initial guess
    itr     = int(opt['itr'])   if 'itr'        in opt.keys() else 100
    beta_k  = opt['beta_k']     if 'beta_k'     in opt.keys() else 1.
    tol     = opt['tol']        if 'tol'        in opt.keys() else 1e-2
    epsilon = opt['epsilon']    if 'epsilon'    in opt.keys() else 0
    rho     = opt['rho']        if 'rho'        in opt.keys() else 1e-2
    delta   = opt['delta']      if 'delta'      in opt.keys() else 5.
    printflag = opt['printflag'] if 'printflag' in opt.keys() else 0
    
    if 'svm_solver' in opt.keys():
        svm_slover = opt['svm_solver']
    else:
        svm_solver = ''
        solvers = cp.installed_solvers()
        if 'MOSEK' in solvers: svm_slover = cp.MOSEK
        if 'GUROBI' in solvers: svm_solver = cp.GUROBI
        if not len(svm_solver): svm_solver = cp.ECOS 
        
    if 'main_solver' in opt.keys():
        main_solver = opt['main_solver']
    else:
        main_solver = ''
        solvers = cp.installed_solvers()
        if 'MOSEK' in solvers: main_solver = cp.MOSEK
        if 'GUROBI' in solvers: main_solver = cp.GUROBI
        if not len(main_solver): main_solver = cp.ECOS 
    
    if printflag:
        print('the solver used for svm is ', svm_slover, 'the solver used for main problem is ', main_solver)
    
    mu_k    = initial_guess['mu']     if 'mu'     in initial_guess.keys() else np.exp((np.log(lambda_lb) + np.log(lambda_ub))/2)
    w_bar_k = initial_guess['w_bar']  if 'w_bar'  in initial_guess.keys() else w_bar_lb*np.ones(n)
    W_k     = initial_guess['W']      if 'W'      in initial_guess.keys() else np.zeros([T, n])
    c_k     = initial_guess['c']      if 'c'      in initial_guess.keys() else np.zeros(T)
    
    # 0-1 Define the T-fold SVM model
    ## 0-1: 1/4 Declare variables and paramters
    W_svm, c = cp.Variable([T, n]), cp.Variable(T)
    lam_svm, w_bar_k_svm = cp.Parameter(nonneg=True), cp.Parameter(n)
    
    ## 0-1: 2/4 Declare svm target
    svm1 = lam_svm*cp.sum_squares(W_svm)
    svm2 = cp.sum( [cp.sum( 
        cp.maximum( 1 - cp.multiply( y[J_trn[i]], (X[J_trn[i], :] @ W_svm[i] - c[i]) ), 0) 
        ) for i in range(T)] ) 
    svm_loss = svm1 + svm2
    
    ## 0-1: 3/4 Declare constraints
    svm_constraints = []
    for t in range(T):
        svm_constraints += [W_svm[t] >= -w_bar_k_svm]
        svm_constraints += [W_svm[t] <=  w_bar_k_svm]
    
    ## 0-1: 4/4 Declare Optimizaiton Problem
    prob_svm = cp.Problem(cp.Minimize(svm_loss), svm_constraints)
    
    
    # 0-2 Define the main optimization model
    ## 0-2: 1/4 Declare variables and paramters
    mu, w_bar = cp.Variable(), cp.Variable(n)
    W, c = cp.Variable([T, n]), cp.Variable(T)
    
    mu_k_30, w_bar_k_30, W_k_30, c_k_30 = cp.Parameter(), cp.Parameter(n), cp.Parameter([T, n]), cp.Parameter(T)
    gamma_2_w, f_k_30, w_bias = cp.Parameter(n), cp.Parameter(), cp.Parameter()
    gamma_2_mu, mu_bias = cp.Parameter(), cp.Parameter()
    beta_k_30 = cp.Parameter(nonneg=True)
    
    ## 0-2: 2/4 Declare target
    F1 = 1/T*1/nVal*cp.sum( [cp.sum( cp.maximum( 1 - cp.multiply( y[J_val[i]], (X[J_val[i], :] @ W[i] - c[i]) ), 0) ) for i in range(T)])
    prox = cp.sum_squares(mu - mu_k_30) + cp.sum_squares(w_bar - w_bar_k_30) + cp.sum_squares(W - W_k_30) + cp.sum_squares(c - c_k_30)
    f1_1 = cp.sum( cp.quad_over_lin(W, 2 * mu) )
    f1_2 = cp.sum( [cp.sum( cp.maximum( 1 - cp.multiply( y[J_trn[i]], (X[J_trn[i], :] @ W[i] - c[i]) ), 0) ) for i in range(T)] ) 
    f1 = beta_k_30 * (f1_1 + f1_2) / nTrn
    f2_lin = gamma_2_w @ w_bar - w_bias + gamma_2_mu * mu - mu_bias
    loss_30 =  F1 + rho/2 * prox + cp.maximum( f1 - f_k_30 - f2_lin - beta_k_30 * epsilon, 0)
    
    ## 0-2: 3/4 Declare constraints
    iP_DCA_constraints = [mu >= 1/lambda_ub] + [mu <= 1/lambda_lb] + [w_bar >= w_bar_lb] + [w_bar <= w_bar_ub]
    for i in range(T):
        iP_DCA_constraints += [W[i] <= w_bar] 
        iP_DCA_constraints += [W[i] >= -w_bar]
        
    ## 0-2: 4/4 Declare Optimizaiton Problem
    prob_30 = cp.Problem(cp.Minimize(loss_30), iP_DCA_constraints)
    
    # 1 main iteration
    for k in range(itr):
        # 1-1: solve the SVMs
        lam_svm.value, w_bar_k_svm.value = .5/mu_k, w_bar_k
        if printflag>1: print("start solving svm")
        prob_svm.solve(solver = svm_slover)
        if printflag>1: print("svm solved")
        
        # 1-2: calculate gamma_2 which actually is gamma_2_w
        gamma_2, f_k = np.zeros(n), prob_svm.value
        for i in range(2*T):
            gamma_2 = gamma_2 - svm_constraints[i].dual_value
        gamma_2 = gamma_2/nTrn
        W_k_m = W_svm.value
        
        # 1-3: solve the approximated subproblem
        mu_k_30.value, w_bar_k_30.value, W_k_30.value, c_k_30.value = mu_k, w_bar_k, W_k, c_k
        beta_k_30.value, f_k_30.value = beta_k, beta_k*f_k/nTrn
        gamma_2_w.value, gamma_2_mu.value = beta_k*gamma_2, -beta_k*np.sum(np.square(W_k_m))/np.square(mu_k)/2/nTrn
        mu_bias.value, w_bias.value = gamma_2_mu.value * mu_k, np.dot(gamma_2_w.value, w_bar_k)
        if printflag>1: print("start solving subproblem")
        prob_30.solve(solver = main_solver)
        if printflag>1: print("subproblem solved")
        mu_k_p, w_bar_k_p, W_k_p, c_k_p = mu.value, w_bar.value, W.value, c.value
        if main_solver == 'SCS': w_bar_k_p = np.maximum(w_bar_k_p, w_bar_lb)
        t_k_p = cp.maximum( f1 - f_k_30 - f2_lin - beta_k*epsilon, 0).value/beta_k

        # 1-4 stopping test
        err = np.sqrt( np.square(mu_k_p - mu_k) + np.sum(np.square(w_bar_k_p - w_bar_k)) 
                     + np.sum(np.square(W_k_p - W_k))   + np.sum(np.square(c_k_p - c_k)) )
        err_rel = err/(1. + np.sqrt(  np.square(mu_k)    + np.sum(np.square(w_bar_k)) 
                                    + np.sum(np.square(W_k)) + np.sum(np.square(c_k)) ) )
        if err_rel < tol and t_k_p < 1e-4:
            if printflag: print('err = ({:.3e}, {:.3e}) Pass the stopping test\n'.format(err_rel, float(t_k_p)))
            mu_k, w_bar_k, W_k, c_k = mu_k_p, w_bar_k_p, W_k_p, c_k_p
            break 
        else:
            if printflag: print('{:4d}-th iteration: err = ({:.3e}, {:.3e})'.format(k+1, err_rel, float(t_k_p)))

        if err_rel < 10*min( 1./beta_k, t_k_p ):
            beta_k = beta_k + delta
            if printflag: print('beta update to ', beta_k)
        
        # 1-5 prepare for next step
        mu_k, w_bar_k, W_k, c_k = mu_k_p, w_bar_k_p, W_k_p, c_k_p
    
    if printflag and (k == itr-1): print('Stop due to the limitation of iteration, the result may be inaccurate.')
    
    obj = dict(F = F1.value)
    var = dict(mu = mu_k, w_bar = w_bar_k, W = W_k, c = c_k, lam = 1./mu_k)
    result = dict(obj = obj, var = var)
    
    return result

#%%
def test(XTrn, yTrn, J_val, J_trn, XTest, yTest, lam, w_bar):
    '''A function for evaluating performance of hyperparameter, which output the cross validation error and test error
    
    Arguments
    ---------
    XTrn, yTrn: features and labels for T-fold SVM
    J_val, J_trn: index set for validation and train in T-fold SVM
    XTest, yTest: data for testing the performance of mu and w_bar
    mu, w_bar: hyper-parameter for test
    
    Return
    ---------
    result = dict(train_err = train_err, cross_err = cross_err, test_err = test_err)
    ---------
    '''
    svm_solver = ''
    solvers = cp.installed_solvers()
    if 'MOSEK' in solvers: svm_slover = cp.MOSEK
    if 'GUROBI' in solvers: svm_solver = cp.GUROBI
    if not len(svm_solver): svm_solver = cp.ECOS 

    T, n, nVal, nTrn = len(J_val), XTrn.shape[1], len(J_val[0]), len(J_trn[0])
    W, c = cp.Variable([T, n]), cp.Variable(T)
    f1 = lam*cp.sum_squares(W)
    f2 = cp.sum( [cp.sum( [cp.maximum( 1 - yTrn[J_trn[i][j]] * (XTrn[J_trn[i][j], :] @ W[i] - c[i]), 0) for j in range(nTrn)] ) for i in range(T)] ) 
    svm_loss = f1 + f2
    
    svm_constraints = []
    for t in range(T):
        svm_constraints += [W[t, i] >= -w_bar[i] for i in range(n)] + [W[t, i] <= w_bar[i] for i in range(n)]
        
    prob_svm = cp.Problem(cp.Minimize(svm_loss), svm_constraints)
    prob_svm.solve(solver = svm_solver)
    
    W_0, c_0 = W.value, c.value
    
    train_err = 1/T*1/nTrn*np.sum( [np.sum([np.maximum( 1 - yTrn[J_trn[i][j]] * (XTrn[J_trn[i][j]] @ W_0[i] - c_0[i]), 0) for j in range(nTrn)] ) for i in range(T)])
    
    cross_err = 1/T*1/nVal*np.sum( [np.sum([np.maximum( 1 - yTrn[J_val[i][j]] * (XTrn[J_val[i][j]] @ W_0[i] - c_0[i]), 0) for j in range(nVal)] ) for i in range(T)])
    
    W, c = cp.Variable(n), cp.Variable()
    f1 = T/(T-1)*lam*cp.sum_squares(W)/2
    f2 = cp.sum( cp.maximum( 1 - cp.multiply(yTrn, XTrn @ W - c), 0) )
    svm_loss = f1 + f2
    
    svm_constraints = [W[i] >= -w_bar[i] for i in range(n)] + [W[i] <= w_bar[i] for i in range(n)]
        
    prob_svm = cp.Problem(cp.Minimize(svm_loss), svm_constraints)
    prob_svm.solve(solver = svm_solver)
    
    W_1, c_1 = W.value, c.value
    test_err = np.mean( np.abs( yTest - np.sign( np.dot(W_1, XTest.toarray().T) - c_1 ) )/2 )
    
    result = dict(train_err = train_err, cross_err = cross_err, test_err = test_err)
    return result
