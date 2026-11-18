# Handwritten Digit Recognition: ANN vs CNN from scratch

Mini project for Deep Learning & Gen AI (Emerging Minor).
Bhagyesh Rajendra Patil | RBT23CS073 | Roll no CS4204

An ANN and a CNN are built with plain NumPy (forward pass, backpropagation, SGD and Adam written by hand) and compared on the scikit-learn 8x8 digits dataset.

## Setup and run

    pip install -r requirements.txt
    python main.py

It runs in well under a minute on a normal laptop and saves all figures in the results folder.

## Files

| File | What it does |
|---|---|
| layers.py | Dense, ReLU, Flatten, Conv2D, MaxPool2D and softmax with cross-entropy |
| optimizers.py | SGD and Adam |
| network.py | Network class, ANN (64-64-32-10) and CNN (Conv 3x3 x8, MaxPool, Dense 64, Dense 10) |
| data.py | Loading, scaling, 80:20 stratified split, noise helper |
| gradcheck.py | Numerical gradient check for the CNN backpropagation |
| train.py | Mini-batch training loop (batch 32, 30 epochs) |
| plots.py | All graphs |
| main.py | Runs everything and prints the results |
| predict_image.py | Predicts a digit from your own image |

## What you get

Gradient check, training logs for ANN + SGD, ANN + Adam and CNN + Adam, a comparison table, classification report, confusion matrix, mistakes, and a noise robustness test.

Exact accuracy changes by a percent or so with the random seed, because the test set has only 360 images (one image is 0.28%).

## Try your own image

    python predict_image.py my_digit.png

Use one digit in dark ink on a plain light background (a photo or a drawing from Paint works). The script trains both models in a few seconds, shrinks your image to 8x8 and prints what the ANN and the CNN predict. Results on your own handwriting can be weaker, because the models were trained on only 1,797 tiny images from a limited group of writers.
