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

3. If using python >= 3.6:

    1. Add the root directory to the PYTHONPATH
        ```
        export PYTHONPATH="${PYTHONPATH}:path/to/BLADE"
        ```
    2. Create a virtual environment with the packages required, that can be found on the file ```required_packages.txt```

4. If using Anaconda, create a virtual environment with the packages required, that can be found on the file ```required_packages.txt```

5. The runnning mode flag that is present in the file ```blade.py``` defines the running mode, with two different possible values:

    - "dev": used when running the code with the values defined on the file, for testing purposes.
    - "run": used when running the code through the command line, with the parameter values indicated by the user.

6. If running in terminal, input the following:

    ```
    blade.py 1 example_data/binomial_1_2 -options
    ```
#### Parameters

The different parameters that can be specified when running the BLADE algorithm are presented in the table below.

| parameter                  | parameter code        |       | default              | description                                                                                                                                                                 |
|----------------------------|---------|-------------|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| filepath                   |         | :heavy_check_mark: | \-\-                 | file path for both the data file \(filepath\+\``\\\_parsed\.csv''\) and the labels file \(filepath \+ ``\\\_labels\.csv''\)                                                 |
| tdbn\\\_parents            | \-tdbnp |             | 1                    | set the number of parent nodes to be considered by the \\acrshort\{tdbn\} algorithm                                                                                         |
| no\\\_rbm                  | \-nrbm  |             | False   | if the user wishes to run the method without the \\acrshort\{rbm\} pre\-processing, he should indicate it by placing this parameter on the input                            |
| test\\\_set\\\_ratio       | \-tsr   |             | 0\.2                 | ratio of the original data to be used for testing                                                                                                                           |
| validation\\\_set\\\_ratio | \-vsr   |             | 0\.2                 | ratio of the pre\-training dataset to be used for validation                                                                                                                |
| batch\\\_size\\\_ratio     | \-bsr   |             | 0\.1                 | ratio to be used when splitting the training set in mini\-batches for the \\acrshort\{cd\} algorithm                                                                        |
| hidden\\\_units            | \-hu    |             | 3                    | number of hidden units to be used in the \\acrshort\{rbm\}s                                                                                                                 |
| epochs                     | \-e     |             | 100                  | defines the number of iterations in the \\acrshort\{cd\}\-1 algorithm                                                                                                       |
| learning\\\_rate           | \-lr    |             | 0\.05                | learning rate for the \\acrshort\{rbm\}                                                                                                                                     |
| weight\\\_decay            | \-wd    |             | 1x10<sup>-4</sup> | weight decay value when training the \\acrshort\{rbm\}                                                                                                                      |
| persistent\\\_cd           | \-pcd   |             | False   | defines the usage of persistent \\acrshort\{cd\}\.                                                                                                                          |
| number\\\_runs             | \-nr    |             | 10                   | since the accuracy has a variance associated, increasing the number of runs of the algorithm ensures that a better approximation of the actual accuracy value is calculated |
| validation\\\_runs         | \-vr    |             | 5                    | number of validation runs                                                                                                                                                   |
| extraction\\\_runs         | \-er    |             | 5                    | number of extractions in each validation cycle                                                                                                                              |
| verbose                    | \-vb    |             | False   | if \\texttt\{True\}, it prints the results of training the \\acrshort\{rbm\} and \\acrshort\{tdbn\} while the \\acrshort\{blade\} algorithm is running                      |
| version                    | \-v     |             | \-\-                 | prints the current version of the \\acrshort\{blade\} implementation                                                                                                        |


### Future work
- 



## References
- tDBN algorithm, extended to create an initial network and calculate the probabilities of a test dataset, developed by [José Monteiro]
- RBM implementation in Python using Pytorch, used as a basis for the implementation here present, and developed by [Gabriel Bianconi], under the MIT License.

<!-- Links -->

[Gabriel Bianconi]: https://github.com/GabrielBianconi/pytorch-rbm

[José Monteiro]: https://github.com/josemonteiro/tDBN

