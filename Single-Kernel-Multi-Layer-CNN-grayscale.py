from framework import (
    InputLayer,
    ConvolutionalLayer,
    MaxPoolLayer,
    FlatteningLayer,
    FullyConnectedLayer,
    SoftmaxLayer,
    CrossEntropy
)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, time

np.random.seed(0)

training_dir = "./cifar-10-training-gray/"
testing_dir = "./cifar-10-testing-gray/"
class_labels = [str(x) for x in range(10)]
img_max = 100
num_images = 10 * img_max

X_train = np.zeros((num_images, 32, 32))
Y_train = np.zeros((num_images, 10))
X_test = np.zeros((num_images, 32, 32))
Y_test = np.zeros((num_images, 10))

def calc_accuracy(H, Y):
    num_correct = 0
    N = Y.shape[0]
    
    for i in range(N):
        max_likelihood_index = np.argmax(H[i])
        if Y[i][max_likelihood_index] == 1:
            num_correct += 1
            
    accuracy = num_correct / N
    return accuracy

train_iter = 0
test_iter = 0

for label in class_labels:
    train_dir = training_dir + label
    test_dir = testing_dir + label
    
    for file in sorted(os.listdir(train_dir))[:img_max]:    
        im = Image.open(train_dir + '/' + file)
        im = np.asmatrix(im.getdata(), dtype=np.float64)
        im = im.reshape((32,32))
        X_train[train_iter] = im
        Y_train[train_iter][-1 - int(label)] = 1
        train_iter += 1
        
    for file in sorted(os.listdir(test_dir))[:img_max]:
        im = Image.open(test_dir + '/' + file)
        im = np.asmatrix(im.getdata(), dtype=np.float64)
        im = im.reshape((32,32))
        X_test[test_iter] = im
        Y_test[test_iter][-1 - int(label)] = 1
        test_iter += 1

L1 = InputLayer(dataIn=X_train)
L2 = ConvolutionalLayer(kSize=3)
L2.setKernels(np.random.randn(3,3) * (np.sqrt(2/9)))
L3 = MaxPoolLayer(size=3, stride=3)
L4 = ConvolutionalLayer(kSize=3)
L4.setKernels(np.random.randn(3,3) * (np.sqrt(2/9)))
L5 = MaxPoolLayer(size=2, stride=2)
L6 = FlatteningLayer()
L7 = FullyConnectedLayer(sizeIn=16, sizeOut=10)
L8 = SoftmaxLayer()
L9 = CrossEntropy()
layers = [L1, L2, L3, L4, L5, L6, L7, L8, L9]

epochs = 100
tolerance = 1e-10
learningRate = 1e-3

trainingCrossEntropyLoss = []
testingCrossEntropyLoss = []

trainingAcc = []
testingAcc = []

tic = time.perf_counter()

for e in range(epochs):    
    # Forward
    trainingH = X_train
    testingH = X_test
    for i in range(len(layers) - 1):
        testingH = layers[i].forward(testingH)
        trainingH = layers[i].forward(trainingH)
    
    # Evalute cross entropy loss and accuracy
    trainingCrossEntropyLoss.append(layers[-1].eval(Y_train, trainingH))
    testingCrossEntropyLoss.append(layers[-1].eval(Y_test, testingH))
    
    trainingAcc.append(calc_accuracy(trainingH, Y_train))
    testingAcc.append(calc_accuracy(testingH, Y_test))
    
    # Backward
    grad = layers[-1].gradient(Y_train, trainingH)
    for i in range(len(layers) - 2, 0, -1):
        newGrad = layers[i].backward(grad)
        
        if isinstance(layers[i], FullyConnectedLayer):
            layers[i].updateWeights(grad, learningRate)
        elif isinstance(layers[i], ConvolutionalLayer):
            layers[i].updateKernels(grad, learningRate)
        
        grad = newGrad
    
    if len(trainingCrossEntropyLoss) > 2 and np.abs(trainingCrossEntropyLoss[-2] - trainingCrossEntropyLoss[-1]) < tolerance:
        print("Difference in cross entropy loss is below threshold. Terminating early...")
        break

toc = time.perf_counter()
print("Time taken (in seconds):", toc-tic)

print("Epochs:", e+1)
print("Learning rate:", learningRate)
print("Tolerance:", tolerance)
print()

plt.plot(trainingCrossEntropyLoss, label = 'Training')
plt.plot(testingCrossEntropyLoss, label = 'Testing')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.title(f"LR={learningRate}, # of images={num_images}")
plt.grid()
plt.legend()
plt.show()

plt.plot(trainingAcc, label = 'Training')
plt.plot(testingAcc, label = 'Testing')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f"LR={learningRate}, # of images={num_images}")
plt.grid()
plt.legend()
plt.show()