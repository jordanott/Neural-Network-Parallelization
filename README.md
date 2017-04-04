# Neural-Network-Parallelization
Using OpenMP to parallelize the training of a neural network

#### Parallezing Options ####  
![AltText](https://github.com/jordanott/Neural-Network-Parallelization/blob/master/images/layer_parallel.png)  

![AltText](https://github.com/jordanott/Neural-Network-Parallelization/blob/master/images/net_parallel.png)  

In this project we implement option two. We create multiple threads training their own instances of the network, with a partition of the data. Then average the weights of all threads networks at the end. You can read my write up [here](https://github.com/jordanott/Neural-Network-Parallelization/blob/master/neural_network_parallelization.pdf).

