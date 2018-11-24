#################################################################
# Code for Preceptron Function & Calculate W's from Training Data
#################################################################

import numpy as np
import os
from scipy import misc


def preceptron_fn(inputs, original_labels, weight_vector, learn_par):
    iteration = 0
    misclass_point = [0, 0]  # any initial value to start while loop
    while len(misclass_point) != 0:
        #print("========================================  New Iteration  ==========================================")
        iteration += 1
        y = []
        calc_t = []

        # Calculate the label "t" based on Y(x)
        for i in range(len(inputs)):
            y = np.dot(np.transpose(weight_vector), np.array(inputs[i]))
            if y >= 0:
                calc_t.append(1)
            else:
                calc_t.append(-1)

        misclass_point = []

        # Compare original label with calculated label & determine misclassified point
        for i in range(len(original_labels)):
            if original_labels[i] != calc_t[i]:
                misclass_point = inputs[i]  # get misclassified point
                t_missclass_point = original_labels[i]  # get t(label) of the misclassified point
                #print("Misclassified Point through w(" + str(iteration) + "):")
                #print(misclass_point)
                #print("The Wrong Label of the misclassified point: ")
                #print(t_missclass_point)
                break

        # Calculate error & get the next W:
        if len(misclass_point) != 0:
            error = np.array(misclass_point) * t_missclass_point
            #print("For iteration number: " + str(iteration) + " , the Error is: " + str(error))
            weight_vector = weight_vector + (learn_par * error)

    print("For iteration number: " + str(iteration) + " , the Error is: 0!" )
    return weight_vector

######################
# Load Training Images
######################
Train_Path='C:\\Users\\chtv2985\\Desktop\\Assignment_1\\Train'
os.chdir(Train_Path)
Training_Labels = np.loadtxt('Training Labels.txt')
files=os.listdir(Train_Path)
files.pop()
files = sorted(files,key=lambda x: int(os.path.splitext(x)[0]))

all_data=[]
for i in files:
    img=misc.imread(i)
    type(img)
    img.shape
    #change dimension to 1 dimensional array instead of (28x28)
    img=img.reshape(784,)
    img=np.append(img,1)
    all_data.append(img)


# Enter the value of Learning Parameter & Initial Weight Vector
eta = 10**-9
w=np.zeros(784)
w=np.append(1, w)
print("Initial Weight Vector W(0)")
print(w)


###########################################################
## Getting decision boundary for 10 lines for different etas
###########################################################

# Changing labels to get class 0 decision boundary
label0 = [1 if n == 0 else -1 for n in Training_Labels]
label1 = [1 if n == 1 else -1 for n in Training_Labels]
label2 = [1 if n == 2 else -1 for n in Training_Labels]
label3 = [1 if n == 3 else -1 for n in Training_Labels]
label4 = [1 if n == 4 else -1 for n in Training_Labels]
label5 = [1 if n == 5 else -1 for n in Training_Labels]
label6 = [1 if n == 6 else -1 for n in Training_Labels]
label7 = [1 if n == 7 else -1 for n in Training_Labels]
label8 = [1 if n == 8 else -1 for n in Training_Labels]
label9 = [1 if n == 9 else -1 for n in Training_Labels]


######################################################################
# Apply Preceptron Function to get weight vectors for different eta's
######################################################################

wsum = []
wtotal = np.array(wsum)
# Weight Vector for Class Zero
w0 = preceptron_fn(all_data, label0, w, eta)
wtotal = np.append(wtotal, w0)

# Weight Vector for Class One
w1 = preceptron_fn(all_data, label1, w, eta)
wtotal = np.append(wtotal, w1)

# Weight Vector for Class Two
w2 = preceptron_fn(all_data, label2, w, eta)
wtotal = np.append(wtotal, w2)

# Weight Vector for Class Three
w3 = preceptron_fn(all_data, label3, w, eta)
wtotal = np.append(wtotal, w3)

# Weight Vector for Class Four
w4 = preceptron_fn(all_data, label4, w, eta)
wtotal = np.append(wtotal, w4)

# Weight Vector for Class Five
w5 = preceptron_fn(all_data, label5, w, eta)
wtotal = np.append(wtotal, w5)

# Weight Vector for Class Six
w6 = preceptron_fn(all_data, label6, w, eta)
wtotal = np.append(wtotal, w6)

# Weight Vector for Class Seven
w7 = preceptron_fn(all_data, label7, w, eta)
wtotal = np.append(wtotal, w7)

# Weight Vector for Class Eight
w8 = preceptron_fn(all_data, label8, w, eta)
wtotal = np.append(wtotal, w8)

# Weight Vector for Class Nine
w9 = preceptron_fn(all_data, label9, w, eta)
wtotal = np.append(wtotal, w9)


#savepath = r'C:\Users\chtv2985\Desktop\Assignment_1\Eta_Files'
#os.chdir(savepath)
np.savetxt(r'C:\Users\chtv2985\Desktop\Assignment_1\Eta_Files\eta_10-9.txt', wtotal)
retrieve_txt = np.loadtxt(r'C:\Users\chtv2985\Desktop\Assignment_1\Eta_Files\eta_10-9.txt')


#print(wtotal)
print("Final W:")
print(type(wtotal))
print(len(wtotal))

print("From Txt File:")
print(type(retrieve_txt))
print(len(retrieve_txt))