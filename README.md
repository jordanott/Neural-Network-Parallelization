# Neural-Network-Parallelization
Using OpenMP to parallelize the training of a neural network

#### Parallezing Options ####  
![AltText](https://github.com/jordanott/Neural-Network-Parallelization/blob/master/Images/distributed_nodes.png)  

![AltText](https://github.com/jordanott/Neural-Network-Parallelization/blob/master/Images/duplicated_net.png)  
Images taken from [Parallelizing Neural Network Training for Cluster Systems](https://www.cs.swarthmore.edu/~newhall/papers/pdcn08.pdf)

In this project we implement option two. We create multiple threads training their own instances of the network, with a partition of the data. Then average the weights of all threads networks at the end.

