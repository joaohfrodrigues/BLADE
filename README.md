## BLADE

--------------------

### Overview
The study of multivariate time-series is applied to a wide range of areas in society, being used to represent all sorts of events and behaviors across time. The BLADE (Boltzmann dynamic network) model aggregates both a dynamic Bayesian networks approach to study time-series data and a restricted Boltzmann machine implementation to derive hidden feature relationships present along the time frame.

### Assumptions
- In this implementation of BLADE, the data is considered to be categorical (features assume one of multiple finite values). Some method of discretization such as symbolic aggregate approximation (SAX) can be used to discretize a real-valued dataset. Some examples of this case were used and are provided in the directory real_data


### How to run the RBM-tDBN algorithm
1. Download a Python distribution

2. Download the repository

```
git clone https://github.com/joaor96/rbm-tdbn
cd rbm-tdbn
```

3. If using python >= 3.6, add the root directory to the PYTHONPATH

```
export PYTHONPATH="${PYTHONPATH}:path/to/BLADE"
```

### Future work
- 



## References
- tDBN algorithm, extended to create an initial network and calculate the probabilities of a test dataset, developed by [José Monteiro]
- RBM implementation in Python using Pytorch, used as a basis for the implementation here present, and developed by [Gabriel Bianconi], under the MIT License.

<!-- Links -->

[Gabriel Bianconi]: https://github.com/GabrielBianconi/pytorch-rbm

[José Monteiro]: https://github.com/josemonteiro/tDBN

