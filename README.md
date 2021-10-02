# iP_DCA
This repository provide a realization of iP_DCA on Bilevel-SVM problem in Python with a simple test demo and a code for reproduce the numerical result of experiments.

The algorithm and the model are presented in the paper [_Difference of convex algorithms for bilevel programs with applications in hyperparameter selection_](https://arxiv.org/pdf/2102.09006.pdf)

## Dependencies
### Python Package Denpency
Beside "usual" packages (`numpy`), iP_DCA is built upon `cvxpy`. 

In consideration of large-scale problem, `scipy` is recommanded for the usage of sparse matrix.

For ease of data reading, `sklearn` is also recommanded.

When `sklearn.datasets.fetch_openml` is used as in `demo_realdata.py`, `pandas` is also required due to the package dependency.

### Solvers 
`cvxpy` contains several open-source solver including `ECOS` and `SCS` we used in our code, but it also provides easy-use interface for other solvers;

Here we recommand researchers to try the commercial solvers for moderate-size problem to attain a better efficiency;

However, for large-scale problem, `SCS` based on `OpenMP` may perform better with the default setting. 

## Usage
To give a try of our algorithm without any local data, one could call 
```
python demo_realdata.py
```
To reproduce the numerical result in paper, one can run the following commands respectively
```
python experiments_realdata.py --solvers GUROBI --repeat_time 20 --data_scale moderate
python experiments_realdata.py --solvers open_source --repeat_time 20 --data_scale moderate
python experiments_realdata.py --solvers open_source --repeat_time 20 --data_scale large
```

