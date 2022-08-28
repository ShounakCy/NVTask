# Using a simple CNN model using tf and keras

## How to run
1. pip install requirements.txt

## Steps Involved
    -Loading the required libraries
    -Using image data genearator augmented the defected dataset
    -Created a train and valid dataset
    -Rescaling it for the model
    -Created an encoder decoder convolution model, followed by maxpooling and activations(ReLu and Sigmoid)
    -Tested with Adadelta and Adam Optimizer, Adam tend to lower to losses, hence used the same.
    -Saved and Loaded the weights after model.fit
    -Testing of the encoder and visualised the restructured image
    -Testing with a correct image
    -MSE above the threshold are defective