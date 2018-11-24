###############################################
# Apply W's of the 10 Eta's on Validation Data
###############################################

import os
import numpy as np
from scipy import misc
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

# Load Validation Images
##########################
Valid_Path='C:\\Users\\chtv2985\\Desktop\\Assignment_1\\Validation'
os.chdir(Valid_Path)
Validation_Labels = np.loadtxt('Validation Labels.txt')
filestest=os.listdir(Valid_Path)
filestest.pop()
filestest = sorted(filestest,key=lambda x: int(os.path.splitext(x)[0]))

all_valid=[]
for i in filestest:
    imgtest=misc.imread(i)
    type(imgtest)
    imgtest.shape
    #change dimension to 1 dimensional array instead of (28x28)
    imgtest=imgtest.reshape(784,)
    imgtest=np.append(imgtest,1)
    all_valid.append(imgtest)

print(len(all_valid))

# Import W's of each eta
##########################
w_eta_0 = np.loadtxt(r'C:\Users\chtv2985\Desktop\Assignment_1\Eta_Files\eta_1.txt')
w_eta_1 = np.loadtxt(r'C:\Users\chtv2985\Desktop\Assignment_1\Eta_Files\eta_10-1.txt')
w_eta_2 = np.loadtxt(r'C:\Users\chtv2985\Desktop\Assignment_1\Eta_Files\eta_10-2.txt')
w_eta_3 = np.loadtxt(r'C:\Users\chtv2985\Desktop\Assignment_1\Eta_Files\eta_10-3.txt')
w_eta_4 = np.loadtxt(r'C:\Users\chtv2985\Desktop\Assignment_1\Eta_Files\eta_10-4.txt')
w_eta_5 = np.loadtxt(r'C:\Users\chtv2985\Desktop\Assignment_1\Eta_Files\eta_10-5.txt')
w_eta_6 = np.loadtxt(r'C:\Users\chtv2985\Desktop\Assignment_1\Eta_Files\eta_10-6.txt')
w_eta_7 = np.loadtxt(r'C:\Users\chtv2985\Desktop\Assignment_1\Eta_Files\eta_10-7.txt')
w_eta_8 = np.loadtxt(r'C:\Users\chtv2985\Desktop\Assignment_1\Eta_Files\eta_10-8.txt')
w_eta_9 = np.loadtxt(r'C:\Users\chtv2985\Desktop\Assignment_1\Eta_Files\eta_10-9.txt')

# Reshape W's to be 10*785
############################
w_eta_0 = w_eta_0.reshape(10, 785)
w_eta_1 = w_eta_1.reshape(10, 785)
w_eta_2 = w_eta_2.reshape(10, 785)
w_eta_3 = w_eta_3.reshape(10, 785)
w_eta_4 = w_eta_4.reshape(10, 785)
w_eta_5 = w_eta_5.reshape(10, 785)
w_eta_6 = w_eta_6.reshape(10, 785)
w_eta_7 = w_eta_7.reshape(10, 785)
w_eta_8 = w_eta_8.reshape(10, 785)
w_eta_9 = w_eta_9.reshape(10, 785)

##################################################
# Apply W's to Validation Images:
##################################################
def calc_confusion_matrix(w_of_eta, valid_image, valid_labels):
    y_eta = []
    y_max_total = []

    for i in range(0, 10):          # Number of W's in each eta file
        for j in range(0, 200):     # Number of Validation Images
            y_w = np.dot(w_of_eta[i], valid_image[j])
            y_eta = np.append(y_eta, y_w)

    # Reshape y_eta to be 200*10
    y_eta = y_eta.reshape(200, 10, order='F')

    # Get the Index of Max Element in each row
    for k in range(0, 200):
        y_max = np.argmax(y_eta[k,:])
        y_max_total = np.append(y_max, y_max_total)

    # Implement While loop to reverse the orders of rows in y_max_total
    y_total_modified = []
    l = 199
    while l != -1:
        y_temp = y_max_total[l]
        y_total_modified = np.append(y_total_modified, y_temp)
        l -= 1

    #print("Now Indexes Corrected!")

    matrix = confusion_matrix(y_total_modified, valid_labels, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Calculate Accuracy:
    diag_sum = np.asarray(matrix)
    eta_acc = np.trace(diag_sum / 2)
    return matrix, eta_acc

# Apply calc_confusion_matrix function to every Eta value with Validation Images:
conf_matrix_eta_0, accuracy_eta_0 = calc_confusion_matrix(w_eta_0, all_valid, Validation_Labels)
print("Confusion Matrix for eta = 1")
print(conf_matrix_eta_0)
print("Accuracy = " + str(accuracy_eta_0) + str("%"))
print("========================================================================================================")

conf_matrix_eta_1, accuracy_eta_1 = calc_confusion_matrix(w_eta_1, all_valid, Validation_Labels)
print("Confusion Matrix for eta = 10^-1")
print(conf_matrix_eta_1)
print("Accuracy = " + str(accuracy_eta_1) + str("%"))
print("========================================================================================================")

conf_matrix_eta_2, accuracy_eta_2 = calc_confusion_matrix(w_eta_2, all_valid, Validation_Labels)
print("Confusion Matrix for eta = 10^-2")
print(conf_matrix_eta_2)
print("Accuracy = " + str(accuracy_eta_2) + str("%"))
print("========================================================================================================")

conf_matrix_eta_3, accuracy_eta_3 = calc_confusion_matrix(w_eta_3, all_valid, Validation_Labels)
print("Confusion Matrix for eta = 10^-3")
print(conf_matrix_eta_3)
print("Accuracy = " + str(accuracy_eta_3) + str("%"))
print("========================================================================================================")

conf_matrix_eta_4, accuracy_eta_4 = calc_confusion_matrix(w_eta_4, all_valid, Validation_Labels)
print("Confusion Matrix for eta = 10^-4")
print(conf_matrix_eta_4)
print("Accuracy = " + str(accuracy_eta_4) + str("%"))
print("========================================================================================================")

conf_matrix_eta_5, accuracy_eta_5 = calc_confusion_matrix(w_eta_5, all_valid, Validation_Labels)
print("Confusion Matrix for eta = 10^-5")
print(conf_matrix_eta_5)
print("Accuracy = " + str(accuracy_eta_5) + str("%"))
print("========================================================================================================")

conf_matrix_eta_6, accuracy_eta_6 = calc_confusion_matrix(w_eta_6, all_valid, Validation_Labels)
print("Confusion Matrix for eta = 10^-6")
print(conf_matrix_eta_6)
print("Accuracy = " + str(accuracy_eta_6) + str("%"))
print("========================================================================================================")

conf_matrix_eta_7, accuracy_eta_7 = calc_confusion_matrix(w_eta_7, all_valid, Validation_Labels)
print("Confusion Matrix for eta = 10^-7")
print(conf_matrix_eta_7)
print("Accuracy = " + str(accuracy_eta_7) + str("%"))
print("========================================================================================================")

conf_matrix_eta_8, accuracy_eta_8 = calc_confusion_matrix(w_eta_8, all_valid, Validation_Labels)
print("Confusion Matrix for eta = 10^-8")
print(conf_matrix_eta_8)
print("Accuracy = " + str(accuracy_eta_8) + str("%"))
print("========================================================================================================")

conf_matrix_eta_9, accuracy_eta_9 = calc_confusion_matrix(w_eta_9, all_valid, Validation_Labels)
print("Confusion Matrix for eta = 10^-9")
print(conf_matrix_eta_9)
print("Accuracy = " + str(accuracy_eta_9) + str("%"))
print("========================================================================================================")


##############################################################################
# Construct a Matrix containing Diagonal Elements of the 10 Confusion Matrices
##############################################################################
get_max = []
for f in range(len(conf_matrix_eta_0)):
    max_temp = max(conf_matrix_eta_0[f,:])
    get_max = np.append(get_max, max_temp)

for f in range(len(conf_matrix_eta_1)):
    max_temp = max(conf_matrix_eta_1[f,:])
    get_max = np.append(get_max, max_temp)

for f in range(len(conf_matrix_eta_2)):
    max_temp = max(conf_matrix_eta_2[f,:])
    get_max = np.append(get_max, max_temp)

for f in range(len(conf_matrix_eta_3)):
    max_temp = max(conf_matrix_eta_3[f,:])
    get_max = np.append(get_max, max_temp)

for f in range(len(conf_matrix_eta_4)):
    max_temp = max(conf_matrix_eta_4[f,:])
    get_max = np.append(get_max, max_temp)

for f in range(len(conf_matrix_eta_5)):
    max_temp = max(conf_matrix_eta_5[f,:])
    get_max = np.append(get_max, max_temp)

for f in range(len(conf_matrix_eta_6)):
    max_temp = max(conf_matrix_eta_6[f,:])
    get_max = np.append(get_max, max_temp)

for f in range(len(conf_matrix_eta_7)):
    max_temp = max(conf_matrix_eta_7[f,:])
    get_max = np.append(get_max, max_temp)

for f in range(len(conf_matrix_eta_8)):
    max_temp = max(conf_matrix_eta_8[f,:])
    get_max = np.append(get_max, max_temp)

for f in range(len(conf_matrix_eta_9)):
    max_temp = max(conf_matrix_eta_9[f,:])
    get_max = np.append(get_max, max_temp)


get_max = get_max.reshape(10, 10, order='C')   # Now each column is representing a class (i.e.: class 0, 1, .......9)
get_max_t = np.transpose(get_max)               # To make each row corresponding to class

get_max_list = get_max_t.tolist()           # Convert it to List
print("Consolidated List:")
print(get_max_list)


# In this function, we will get the value of indexes having largest element in each row (class)
# In case of several elements having the same value, we will select the largest index
# The larger index we select means we're selecting the smaller Eta which will provide better accuracy (ex: index 6 means eta=10^-6)
# for example for a certain class, if index [3] & index [7] both are equal to the same accuracy, we will choose index [7] because it corresponds
# to the smaller eta 10^-7.
# Normally smaller values of learning parameters consume more time in iterations, but usually provide better accuracy.
def max_elements(seq):
    ''' Return list of position(s) of largest element '''
    max_indices = []
    if seq:
        max_val = seq[0]
        for i,val in ((i,val) for i,val in enumerate(seq) if val >= max_val):
            if val == max_val:
                max_indices.append(i)
            else:
                max_val = val
                max_indices = [i]

    return max(max_indices)


max_acc = []
for n in range(len(get_max_list)):
    max_buff = max_elements(get_max_list[n])
    max_acc = np.append(max_acc, max_buff)

print("List of Maximum Accuracies Corresponding to Eta values:")
print(max_acc)

# Accurate W shall be established on the best values of Eta's obtained above calculation "max_acc"
accurate_w = [w_eta_9[0], w_eta_6[1], w_eta_5[2], w_eta_9[3], w_eta_7[4], w_eta_9[5], w_eta_7[6], w_eta_9[7], w_eta_9[8], w_eta_9[9]]


#####################################################################################
# Load Test Images to calculate the confusion matrix using "accurate_w"
#####################################################################################
Test_Path='C:\\Users\\chtv2985\\Desktop\\Assignment_1\\Test'
os.chdir(Test_Path)
Testing_Labels = np.loadtxt('Test Labels.txt')
filestest=os.listdir(Test_Path)
filestest.pop()
filestest = sorted(filestest,key=lambda x: int(os.path.splitext(x)[0]))

all_test=[]
for i in filestest:
    imgtest=misc.imread(i)
    type(imgtest)
    imgtest.shape
    #change dimension to 1 dimensional array instead of (28x28)
    imgtest=imgtest.reshape(784,)
    imgtest=np.append(imgtest,1)
    all_test.append(imgtest)

conf_matrix_best_eta, accuracy_best_eta = calc_confusion_matrix(np.array(accurate_w), all_test, Testing_Labels)
print("Confusion Matrix for Best Selected Eta's")
print(conf_matrix_best_eta)
print("Accuracy = " + str(accuracy_best_eta) + str("%"))
print("========================================================================================================")


###############################################################################
# Save Confusion Matrix as jpg image
###############################################################################
fig_conf_matrix_best_eta, ax = plot_confusion_matrix(conf_mat=conf_matrix_best_eta)
plt.title(label='Confusion Matrix (Best Eta)', fontdict=None, loc='center', pad=None)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.show()
fig_conf_matrix_best_eta.savefig(r'C:\Users\chtv2985\Desktop\Assignment_1\Confusion Matrices (Validation Phase)\Confusion-b.jpg')
