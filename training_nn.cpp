/*
Author: Jordan Ott
Date: 03/04/2017
Description:
This is a single layer neural network for the mnist data set
Copies of this network will be made in threads to parallelize the training of the network

This code is adapted from: https://github.com/HyTruongSon/Neural-Network-MNIST-CPP
*/

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
// n2 = Number of hidden neurons
// n3 = Number of output neurons
// epochs = Number of iterations for back-propagation algorithm
// learning_rate = Learing rate
// momentum = Momentum (heuristics to optimize back-propagation algorithm)
// epsilon = Epsilon, no more iterations if the learning error is smaller than epsilon

const int n1 = width * height; // = 784, without bias neuron 
const int n2 = 128; 
const int n3 = 10; // Ten classes: 0 - 9
const int epochs = 512;
const double learning_rate = 1e-3;
const double momentum = 0.9;
const double epsilon = 1e-3;

// From layer 1 to layer 2. Or: Input layer - Hidden layer
double *w1[n1], *delta1[n1], *out1;

// From layer 2 to layer 3. Or; Hidden layer - Output layer
double *w2[n2], *delta2[n2], *in2, *out2, *theta2;

// Layer 3 - Output layer
double *in3, *out3, *theta3;
double expected[n3];

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
            cout << number << endl;
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

void init_array() {
	// Layer 1 - Layer 2 = Input layer - Hidden layer
    for (int i =0; i < n1; ++i) {
        w1[i] = new double [n2];
        delta1[i] = new double [n2];
    }
    
    out1 = new double [n1];

	// Layer 2 - Layer 3 = Hidden layer - Output layer
    for (int i =0; i < n2; ++i) {
        w2[i] = new double [n3];
        delta2[i] = new double [n3];
    }
    
    in2 = new double [n2];
    out2 = new double [n2];
    theta2 = new double [n2];

	// Layer 3 - Output layer
    in3 = new double [n3];
    out3 = new double [n3];
    theta3 = new double [n3];
    
    // Initialization for weights from Input layer to Hidden layer
    for (int i =0; i < n1; ++i) {
        for (int j =0; j < n2; ++j) {
            int sign = rand() % 2;

            // Another strategy to randomize the weights - quite good 
            // w1[i][j] = (double)(rand() % 10) / (10 * n2);
            
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
            // w2[i][j] = (double)(rand() % 6) / 10.0;

            w2[i][j] = (double)(rand() % 10) / (10.0 * n3);
            if (sign == 1) {
				w2[i][j] = - w2[i][j];
			}
        }
	}
}

// +------------------+
// | Sigmoid function |
// +------------------+

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// +------------------------------------+
// | Forward process - One Hidden Layer |
// +------------------------------------+

void forward_pass() {
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
}

// +---------------+
// | Norm L2 error |
// +---------------+

double square_error(){
    double res = 0.0;
    for (int i =0; i < n3; ++i) {
        res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
	}
    res *= 0.5;
    return res;
}

// +----------------------------+
// | Back Propagation Algorithm |
// +----------------------------+

void back_propagation() {
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
        for (int j = 1 ; j < n2 ; j++ ) {
            delta1[i][j] = (learning_rate * theta2[j] * out1[i]) + (momentum * delta1[i][j]);
            w1[i][j] += delta1[i][j];
        }
	}
}

// +-------------------------------------------------+
// | Learning process: Perceptron - Back propagation |
// +-------------------------------------------------+

int learning_process() {
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
        forward_pass();
        back_propagation();
        if (square_error() < epsilon) {
			return i;
		}
    }
    return epochs;
}

// +--------------------------------------------------------------+
// | Reading input - gray scale image and the corresponding label |
// +--------------------------------------------------------------+

void input(int index) {
	// Reading image
    for (int i =0; i < 784; ++i) {

        out1[i] = mnist_training_data[index][i];
        
	}
    for (int i = 0; i < n3; ++i) {
        expected[i] = mnist_label_data[index][i];
    }
}

// +------------------------+
// | Saving weights to file |
// +------------------------+

void write_matrix(string file_name) {
    ofstream file(file_name.c_str(), ios::out);
	
	// Input layer - Hidden layer
    for (int i =0; i < n1; ++i) {
        for (int j =0; j < n2; ++j) {
			file << w1[i][j] << " ";
		}
		file << endl;
    }
	
	// Hidden layer - Output layer
    for (int i =0; i < n2; ++i) {
        for (int j =0; j < n3; ++j) {
			file << w2[i][j] << " ";
		}
        file << endl;
    }
	
	file.close();
}


void print_info(int index){
    for(int j=0;j<784;++j){
        cout << mnist_training_data[index][j];
        if((j+1) %28 == 0){
            cout << endl;
        }
        
    }
    cout << endl;
    for (int j = 0; j < 10; ++j)
    {
        cout << mnist_label_data[index][j];            
    }
    cout << endl;
    for (int i = 0; i < 10; ++i)
    {
        cout << out3[i];
    }
    cout << endl;
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
    init_array();
    clock_t begin = clock();

    for (int sample =0; sample < nTraining; ++sample) {
        cout << "Sample " << sample << endl;
        
        // Getting (image, label)
        input(sample);
		
		// Learning process: Perceptron (Forward procedure) - Back propagation
        int nIterations = learning_process();

		// Write down the squared error
		cout << "No. iterations: " << nIterations << endl;
        printf("Error: %0.6lf\n\n", square_error());
        report << "Sample " << sample << ": No. iterations = " << nIterations << ", Error = " << square_error() << endl;
		
		// Save the current network (weights)
		if (sample % 50 == 0) {
            print_info(sample);
        }
		
    }
	clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Elapsed time: " << elapsed_secs /60 << endl;
	// Save the final network
    write_matrix(model_fn);
    
    report.close();
    
    return 0;
}
