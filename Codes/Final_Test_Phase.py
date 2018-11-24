########################################
# Apply W's of the 10 Eta's on Test Data
#########################################

import os
import numpy as np
from scipy import misc
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

#load test images
#################
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
# Apply W's to Test Images:
##################################################
def calc_confusion_matrix(w_of_eta, test_image, test_labels):
    y_eta = []
    y_max_total = []

    for i in range(0, 10):          # Number of W's in each eta file
        for j in range(0, 200):     # Number of Test Images
            y_w = np.dot(w_of_eta[i], test_image[j])
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

    # Compute confusion matrix to evaluate the accuracy of a classification
    matrix = confusion_matrix(y_total_modified, test_labels, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Calculate accuracy
    diag_sum = np.asarray(matrix)
    eta_acc = np.trace(diag_sum / 2)
    #print("Accuracy = " + str(eta_acc) + "%")
    return matrix, eta_acc


conf_matrix_eta_0, accuracy_eta_0 = calc_confusion_matrix(w_eta_0, all_test, Testing_Labels)
print("Confusion Matrix for eta = 1")
print(conf_matrix_eta_0)
print("Accuracy = " + str(accuracy_eta_0) + str("%"))
print("========================================================================================================")

conf_matrix_eta_1, accuracy_eta_1 = calc_confusion_matrix(w_eta_1, all_test, Testing_Labels)
print("Confusion Matrix for eta = 10^-1")
print(conf_matrix_eta_1)
print("Accuracy = " + str(accuracy_eta_1) + str("%"))
print("========================================================================================================")

conf_matrix_eta_2, accuracy_eta_2 = calc_confusion_matrix(w_eta_2, all_test, Testing_Labels)
print("Confusion Matrix for eta = 10^-2")
print(conf_matrix_eta_2)
print("Accuracy = " + str(accuracy_eta_2) + str("%"))
print("========================================================================================================")

conf_matrix_eta_3, accuracy_eta_3 = calc_confusion_matrix(w_eta_3, all_test, Testing_Labels)
print("Confusion Matrix for eta = 10^-3")
print(conf_matrix_eta_3)
print("Accuracy = " + str(accuracy_eta_3) + str("%"))
print("========================================================================================================")

conf_matrix_eta_4, accuracy_eta_4 = calc_confusion_matrix(w_eta_4, all_test, Testing_Labels)
print("Confusion Matrix for eta = 10^-4")
print(conf_matrix_eta_4)
print("Accuracy = " + str(accuracy_eta_4) + str("%"))
print("========================================================================================================")

conf_matrix_eta_5, accuracy_eta_5 = calc_confusion_matrix(w_eta_5, all_test, Testing_Labels)
print("Confusion Matrix for eta = 10^-5")
print(conf_matrix_eta_5)
print("Accuracy = " + str(accuracy_eta_5) + str("%"))
print("========================================================================================================")

conf_matrix_eta_6, accuracy_eta_6 = calc_confusion_matrix(w_eta_6, all_test, Testing_Labels)
print("Confusion Matrix for eta = 10^-6")
print(conf_matrix_eta_6)
print("Accuracy = " + str(accuracy_eta_6) + str("%"))
print("========================================================================================================")

conf_matrix_eta_7, accuracy_eta_7 = calc_confusion_matrix(w_eta_7, all_test, Testing_Labels)
print("Confusion Matrix for eta = 10^-7")
print(conf_matrix_eta_7)
print("Accuracy = " + str(accuracy_eta_7) + str("%"))
print("========================================================================================================")

conf_matrix_eta_8, accuracy_eta_8 = calc_confusion_matrix(w_eta_8, all_test, Testing_Labels)
print("Confusion Matrix for eta = 10^-8")
print(conf_matrix_eta_8)
print("Accuracy = " + str(accuracy_eta_8) + str("%"))
print("========================================================================================================")

conf_matrix_eta_9, accuracy_eta_9 = calc_confusion_matrix(w_eta_9, all_test, Testing_Labels)
print("Confusion Matrix for eta = 10^-9")
print(conf_matrix_eta_9)
print("Accuracy = " + str(accuracy_eta_9) + str("%"))
print("========================================================================================================")


###############################################################################
# Save Figures for Confusion Matrices as jpg image
###############################################################################
fig_conf_matrix_eta_0, ax = plot_confusion_matrix(conf_mat=conf_matrix_eta_0)
plt.title(label='Confusion Matrix (eta= 1)', fontdict=None, loc='center', pad=None)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
#plt.show()
fig_conf_matrix_eta_0.savefig(r'C:\Users\chtv2985\Desktop\Assignment_1\Confusion Matrices (Test Phase)\Confusion-0.jpg')

fig_conf_matrix_eta_1, ax = plot_confusion_matrix(conf_mat=conf_matrix_eta_1)
plt.title(label='Confusion Matrix (eta= 10^-1)', fontdict=None, loc='center', pad=None)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
#plt.show()
fig_conf_matrix_eta_1.savefig(r'C:\Users\chtv2985\Desktop\Assignment_1\Confusion Matrices (Test Phase)\Confusion-1.jpg')

fig_conf_matrix_eta_2, ax = plot_confusion_matrix(conf_mat=conf_matrix_eta_2)
plt.title(label='Confusion Matrix (eta= 10^-2)', fontdict=None, loc='center', pad=None)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
#plt.show()
fig_conf_matrix_eta_2.savefig(r'C:\Users\chtv2985\Desktop\Assignment_1\Confusion Matrices (Test Phase)\Confusion-2.jpg')

fig_conf_matrix_eta_3, ax = plot_confusion_matrix(conf_mat=conf_matrix_eta_3)
plt.title(label='Confusion Matrix (eta= 10^-3)', fontdict=None, loc='center', pad=None)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
#plt.show()
fig_conf_matrix_eta_3.savefig(r'C:\Users\chtv2985\Desktop\Assignment_1\Confusion Matrices (Test Phase)\Confusion-3.jpg')

fig_conf_matrix_eta_4, ax = plot_confusion_matrix(conf_mat=conf_matrix_eta_4)
plt.title(label='Confusion Matrix (eta= 10^-4)', fontdict=None, loc='center', pad=None)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
#plt.show()
fig_conf_matrix_eta_4.savefig(r'C:\Users\chtv2985\Desktop\Assignment_1\Confusion Matrices (Test Phase)\Confusion-4.jpg')

fig_conf_matrix_eta_5, ax = plot_confusion_matrix(conf_mat=conf_matrix_eta_5)
plt.title(label='Confusion Matrix (eta= 10^-5)', fontdict=None, loc='center', pad=None)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
#plt.show()
fig_conf_matrix_eta_5.savefig(r'C:\Users\chtv2985\Desktop\Assignment_1\Confusion Matrices (Test Phase)\Confusion-5.jpg')

fig_conf_matrix_eta_6, ax = plot_confusion_matrix(conf_mat=conf_matrix_eta_6)
plt.title(label='Confusion Matrix (eta= 10^-6)', fontdict=None, loc='center', pad=None)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
#plt.show()
fig_conf_matrix_eta_6.savefig(r'C:\Users\chtv2985\Desktop\Assignment_1\Confusion Matrices (Test Phase)\Confusion-6.jpg')

fig_conf_matrix_eta_7, ax = plot_confusion_matrix(conf_mat=conf_matrix_eta_7)
plt.title(label='Confusion Matrix (eta= 10^-7)', fontdict=None, loc='center', pad=None)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
#plt.show()
fig_conf_matrix_eta_7.savefig(r'C:\Users\chtv2985\Desktop\Assignment_1\Confusion Matrices (Test Phase)\Confusion-7.jpg')

fig_conf_matrix_eta_8, ax = plot_confusion_matrix(conf_mat=conf_matrix_eta_8)
plt.title(label='Confusion Matrix (eta= 10^-8)', fontdict=None, loc='center', pad=None)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
#plt.show()
fig_conf_matrix_eta_8.savefig(r'C:\Users\chtv2985\Desktop\Assignment_1\Confusion Matrices (Test Phase)\Confusion-8.jpg')

fig_conf_matrix_eta_9, ax = plot_confusion_matrix(conf_mat=conf_matrix_eta_9)
plt.title(label='Confusion Matrix (eta= 10^-9)', fontdict=None, loc='center', pad=None)
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
#plt.show()
fig_conf_matrix_eta_9.savefig(r'C:\Users\chtv2985\Desktop\Assignment_1\Confusion Matrices (Test Phase)\Confusion-9.jpg')
