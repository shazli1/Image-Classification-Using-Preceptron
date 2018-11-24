#Project Scope

First Requirement:
==================
It was required to design a Perceptron-based classification algorithm that can recognize scanned images of the 10 digits (0 to 9).'\n'
The data set three folders: “Train”, “Validation” and “Test”. The “Train” folder contains 240 images for each digit, while each of the “Validation” and “Test” folders contain 20 images for each digit.
The images in the “Train” folder should be used to train a classifier for each digit using the Perceptron Discriminant Function for k-classes. The folder contains a file named “Training Labels.txt” which includes the labels of the 2400 images in order.
You need to train the classifiers using each of the following values for the learning rate η = 1, 10^-1, 10^-2, 10^-3, 10^-4, 10^-5, 10^-6, 10^-7, 10^-8, 10^-9.
For all Perceptrons, use an initial weight vector that has 1 as the first component (w1) and the rest are zeros.
After the classifiers are trained, test each classifier using the images given in the “Test” folder. The folder also contains a text file named “Test Labels.txt” which include the labels of the 200 images in order.

Second Requirement:
===================
Use the data in the “Validation” folder to find the value of η that achieves the best accuracy for each digit classifier.
Use the best classifier of each digit to classify the data in the “Test” folder. The “Validation” folder also contains a text file named “Validation Labels.txt” which include the labels of the 200 images in order.

Provided Deliverables:
======================
1- "Codes" Folder containing 3 .py codes: for Perceptron Function, applying Perceptron Function for Test images, and applying Perceptron Function on Validation images to select the most accurate learning parameter.
2- "Data Set" Folder containing: Test, Train, and Validation images respectively.
3- "Weight Vectors" Folder containing: 10 txt files including the output weight vectors contained through the Perceptron Function code corresponding to the ten requested values of learning parameter (η = 1, 10^-1, 10^-2, 10^-3, 10^-4, 10^-5, 10^-6, 10^-7, 10^-8, 10^-9).
4-  "Confusion Matrices" containing two folders: confusion matrices from Test phase and Validation phase respectively.

Remarks:
========
Largest accuracy reached in the first requirement is 87% through using learning parameter η = 10^-3.
In the second requirement, in case we have several values of η providing the same classification accuracy of certain class, I've chosen the smallest η value for more general accurate results as it will be moving with smaller steps in iterations (however, it will consume more processing time indeed).
