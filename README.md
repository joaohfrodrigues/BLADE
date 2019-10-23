## RBM-tDBN

--------------------

### Overview
The study of multivariate time-series is applied to a wide range of areas in society, being used to represent all sorts of events and behaviors across time. The RBM-tDBN model aggregates both a dynamic Bayesian networks approach to study time-series data and a restricted Boltzmann machine implementation to derive hidden feature relationships present along the time frame.

### Assumptions
- In this implementation of the RBM-tDBN, it is assumed that the data has a Markov property of order 1, meaning that the feature values at time *t* are only dependent of their values at *t-1*.

- The system is considered to be stationary, maintaining the same transition probabilities across all the time frame.


### How to run the RBM-tDBN algorithm
1. Download a Python distribution

2. Download the repository

```
git clone https://github.com/joaor96/rbm-tdbn
cd rbm-tdbn
```

3. If using python >= 3.6, add the root directory to the PYTHONPATH

```
export PYTHONPATH="${PYTHONPATH}:path/to/rbm-tdbn"
```

### Future work
- 



## References
- tDBN algorithm, extended to create an initial network and calculate the probabilities of a test dataset, developed by [José Monteiro]
- RBM implementation in Python using Pytorch, used as a basis for the implementation here present, and developed by [Gabriel Bianconi], under the MIT License.

<!-- Links -->

[Gabriel Bianconi]: https://github.com/GabrielBianconi/pytorch-rbm

[José Monteiro]: https://github.com/josemonteiro/tDBN

