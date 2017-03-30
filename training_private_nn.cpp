/*
Author: Jordan Ott
Date: 03/04/2017
Description:
This is a single layer neural network for the mnist data set
Copies of this network will be made in threads to parallelize the training of the network

This code is adapted from: https://github.com/HyTruongSon/Neural-Network-MNIST-CPP
*/
#include <omp.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>
#include <ctime>

using namespace std;

#define NUM_THREADS 12
// Training image file name
const string training_image_fn = "mnist/train-images.idx3-ubyte";

// Training label file name
const string training_label_fn = "mnist/train-labels.idx1-ubyte";

// Weights file name
const string model_fn = "model-neural-network.dat";

// Report file name
const string report_fn = "training-report.dat";

// Number of training samples
const int nTraining = 50000;

// Image size in MNIST database
const int width = 28;
const int height = 28;

// n1 = Number of input neurons
// n2 = Number of hidden neurons ResearchPriority1
// n3 = Number of output neurons
// epochs = Number of iterations for back-propagation algorithm
// learning_rate = Learing rate
// momentum = Momentum (heuristics to optimize back-propagation algorithm)
// epsilon = Epsilon, no more iterations if the learning error is smaller than epsilon

const int n1 = 784; // = 784, without bias neuron 
const int n2 = 128; 
const int n3 = 10; // Ten classes: 0 - 9
const int epochs = 512;
const double learning_rate = 1e-3;
const double momentum = 0.9;
const double epsilon = 1e-3;

// From layer 1 to layer 2. Or: Input layer - Hidden layer
double *global_w1[n1];

// From layer 2 to layer 3. Or; Hidden layer - Output layer
double *global_w2[n2];



double mnist_training_data[60000][784];
double mnist_label_data[60000][10];
// File stream to write down a report
ofstream report;

// +--------------------+
// | About the software |
// +--------------------+

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}
void Read_MNIST_training(int NumberOfImages, int DataOfAnImage)
{
    ifstream file ("mnist/train-images.idx3-ubyte",ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    if(temp == 0){
                        mnist_training_data[i][(n_rows*r)+c]= (double)temp;
                    }
                    else{
                        mnist_training_data[i][(n_rows*r)+c]=1;
                    }
                    
                }
            }
        }
    }
    file.close();
}
void Read_MNIST_label(int number_of_images,int i)
{
    ifstream file ("mnist/train-labels.idx1-ubyte",ios::binary);

    if (file.is_open())
    {
        int num = 0;
        int magic_number = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &num,sizeof(num));
        num = ReverseInt(num);

        for(int img = 0; img < number_of_images; ++img)
        {
            unsigned char temp = 0;

            file.read((char*) &temp, sizeof(temp));

            int number = (double)temp;
            
            for (int i =0; i < n3; ++i) {
                mnist_label_data[img][i] = 0.0;
            }
            mnist_label_data[img][number] = 1.0;

        }

    }
}
void about() {
	// Details
	cout << "**************************************************" << endl;
	cout << "*** Training Neural Network for MNIST database ***" << endl;
	cout << "**************************************************" << endl;
	cout << endl;
	cout << "No. input neurons: " << n1 << endl;
	cout << "No. hidden neurons: " << n2 << endl;
	cout << "No. output neurons: " << n3 << endl;
	cout << endl;
	cout << "No. iterations: " << epochs << endl;
	cout << "Learning rate: " << learning_rate << endl;
	cout << "Momentum: " << momentum << endl;
	cout << "Epsilon: " << epsilon << endl;
	cout << endl;
	cout << "Training image data: " << training_image_fn << endl;
	cout << "Training label data: " << training_label_fn << endl;
	cout << "No. training sample: " << nTraining << endl << endl;
}

// +-----------------------------------+
// | Memory allocation for the network |
// +-----------------------------------+


void init_global(){
	for (int i =0; i < n1; ++i) {
        global_w1[i] = new double [n2];
    }

	// Layer 2 - Layer 3 = Hidden layer - Output layer
    for (int i =0; i < n2; ++i) {
        global_w2[i] = new double [n3];
    }
	// Initialization for weights from Input layer to Hidden layer
    for (int i =0; i < n1; ++i) {
        for (int j =0; j < n2; ++j) {
            
            global_w1[i][j] = 0;
        }
	}
	
	// Initialization for weights from Hidden layer to Output layer
    for (int i =0; i < n2; ++i) {
        for (int j =0; j < n3; ++j) {
            
            global_w2[i][j] = 0;
        }
	}
}


// +------------------+
// | Sigmoid function |
// +------------------+

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}



// +------------------------+
// | Saving weights to file |
// +------------------------+

void write_matrix(string file_name) {
    ofstream file(file_name.c_str(), ios::out);
	
	// Input layer - Hidden layer
    for (int i =0; i < n1; ++i) {
        for (int j =0; j < n2; ++j) {
			file << global_w1[i][j] << " ";
		}
		file << endl;
    }
	
	// Hidden layer - Output layer
    for (int i =0; i < n2; ++i) {
        for (int j =0; j < n3; ++j) {
			file << global_w2[i][j] << " ";
		}
        file << endl;
    }
	
	file.close();
}

// +--------------+
// | Main Program |
// +--------------+

int main(int argc, char *argv[]) {
    Read_MNIST_training(50000,784);
    Read_MNIST_label(50000,10);
    
    about();

	
    report.open(report_fn.c_str(), ios::out);
		
	// Neural Network Initialization
    clock_t begin = clock();
	init_global();
    int sample_per_thread = nTraining / NUM_THREADS;
    #pragma omp parallel num_threads(NUM_THREADS) default(none) shared(cout,mnist_label_data,mnist_training_data,sample_per_thread,global_w2,global_w1)
    {
        // From layer 1 to layer 2. Or: Input layer - Hidden layer
        double w1[784][128], delta1[784][128], out1[784];

        // From layer 2 to layer 3. Or; Hidden layer - Output layer
        double w2[128][10], delta2[128][10], in2[128], out2[128], theta2[128];

        // Layer 3 - Output layer
        double in3[10], out3[10], theta3[10];
        double expected[10];


// ******************* INIT ARRAY *******************
        for (int i =0; i < n1; ++i) {
            for (int j =0; j < n2; ++j) {
                int sign = rand() % 2;
                 
                w1[i][j] = (double)(rand() % 6) / 10.0;
                if (sign == 1) {
                    w1[i][j] = - w1[i][j];
                }
            }
        }
        // Initialization for weights from Hidden layer to Output layer
        for (int i =0; i < n2; ++i) {
            for (int j =0; j < n3; ++j) {
                int sign = rand() % 2;
                
                // Another strategy to randomize the weights - quite good 
                 //w2[i][j] = (double)(rand() % 6) / 10.0;

                w2[i][j] = (double)(rand() % 10) / (10.0 * n3);
                if (sign == 1) {
                    w2[i][j] = - w2[i][j];
                }
            }
        }


        #pragma omp for 
        for (int sample =0; sample < 50000; ++sample) {
            //cout << omp_get_thread_num() << " : " << omp_get_num_threads() << endl;
            // GET SAMPLE
            for (int i =0; i < 784; ++i) {
                out1[i] = mnist_training_data[sample][i];
            }
            for (int i = 0; i < n3; ++i) {
                expected[i] = mnist_label_data[sample][i];
            }


    		// Learning process
            for (int i =0; i < n1; ++i) {
                for (int j =0; j < n2; ++j) {
                    delta1[i][j] = 0.0;
                }
            }

            for (int i =0; i < n2; ++i) {
                for (int j =0; j < n3; ++j) {
                    delta2[i][j] = 0.0;
                }
            }

            for (int i =0; i < epochs; ++i) {
                // forward 
                for (int i =0; i < n2; ++i) {
                    in2[i] = 0.0;
                }

                for (int i =0; i < n3; ++i) {
                    in3[i] = 0.0;
                }

                for (int i =0; i < n1; ++i) {
                    for (int j =0; j < n2; ++j) {
                        in2[j] += out1[i] * w1[i][j];
                    }
                }

                for (int i =0; i < n2; ++i) {
                    out2[i] = sigmoid(in2[i]);
                }

                for (int i =0; i < n2; ++i) {
                    for (int j =0; j < n3; ++j) {
                        in3[j] += out2[i] * w2[i][j];
                    }
                }

                for (int i =0; i < n3; ++i) {
                    out3[i] = sigmoid(in3[i]);
                }

                // back prop
                double sum;
                for (int i =0; i < n3; ++i) {
                    theta3[i] = out3[i] * (1 - out3[i]) * (expected[i] - out3[i]);
                }

                for (int i =0; i < n2; ++i) {
                    sum = 0.0;
                    for (int j =0; j < n3; ++j) {
                        sum += w2[i][j] * theta3[j];
                    }
                    theta2[i] = out2[i] * (1 - out2[i]) * sum;
                }

                for (int i =0; i < n2; ++i) {
                    for (int j =0; j < n3; ++j) {
                        delta2[i][j] = (learning_rate * theta3[j] * out2[i]) + (momentum * delta2[i][j]);
                        w2[i][j] += delta2[i][j];
                        
                    }
                }

                for (int i =0; i < n1; ++i) {
                    for (int j = 0 ; j < n2 ; j++ ) {
                        delta1[i][j] = (learning_rate * theta2[j] * out1[i]) + (momentum * delta1[i][j]);
                        w1[i][j] += delta1[i][j];
                        
                    }
                }
            }
            if((sample + 1) % sample_per_thread == 0)
            {
                for (int i =0; i < n1; ++i) {
                    for (int j = 0; j < n2 ; j++ ) {
                        #pragma omp critical
                        {
                            global_w1[i][j] += w1[i][j];
                        }
                    }
                }
                for (int i =0; i < n2; ++i) {
                    for (int j = 0; j < n3 ; j++ ) {
                        #pragma omp critical
                        {
                            global_w2[i][j] += w2[i][j];
                        }
                    }
                }
            }
            
        }
    }
	clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    for (int i =0; i < n1; ++i) {
        for (int j = 0; j < n2 ; j++ ) {
           // global_w1[i][j] /= NUM_THREADS;
            
        }
    }
    for (int i =0; i < n2; ++i) {
        for (int j = 0; j < n3 ; j++ ) {
          //  global_w2[i][j] /= NUM_THREADS;
            
        }
    }

	// Save the final network
    write_matrix(model_fn);
    report << "Elapsed time: " << elapsed_secs /60 << endl;
    report.close();
    
    return 0;
}

