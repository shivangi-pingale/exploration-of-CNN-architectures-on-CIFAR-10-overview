from framework import (
    InputLayer,
    ConvolutionalLayer,
    Convolutional3DLayer,
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

training_dir = "./cifar-10-training-color/"
testing_dir = "./cifar-10-testing-color/"
class_labels = [str(x) for x in range(10)]
img_max = 500
num_images = 10 * img_max

X_train = np.zeros((num_images, 32, 32,3))
Y_train = np.zeros((num_images, 10))
X_test = np.zeros((num_images, 32, 32,3))
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
        im = np.array(im, dtype=np.float64)
        X_train[train_iter] = im
        Y_train[train_iter][-1 - int(label)] = 1
        train_iter += 1
        
    for file in sorted(os.listdir(test_dir))[:img_max]:
        im = Image.open(test_dir + '/' + file)
        im = np.array(im, dtype=np.float64)
        X_test[test_iter] = im
        Y_test[test_iter][-1 - int(label)] = 1
        test_iter += 1

L1 = InputLayer(dataIn=X_train)
L2 = Convolutional3DLayer(kSize=3,numKernels=1)
L3 = MaxPoolLayer(size=2, stride=2)
L4 = FlatteningLayer()
L5 = FullyConnectedLayer(sizeIn=225, sizeOut=10)
L6 = SoftmaxLayer()
L7 = CrossEntropy()
layers = [L1, L2, L3, L4, L5, L6, L7]

epochs = 100
tolerance = 1e-10
learningRate = 1e-2
l2_lambda = 1e-4 

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
    
    training_loss = layers[-1].eval(Y_train, trainingH)
    testing_loss = layers[-1].eval(Y_test, testingH)
    trainingCrossEntropyLoss.append(training_loss)
    testingCrossEntropyLoss.append(testing_loss)
    
    trainingAcc.append(calc_accuracy(trainingH, Y_train))
    testingAcc.append(calc_accuracy(testingH, Y_test))
    
    if (e + 1) % 10 == 0:
        print(f"Epoch {e+1}/{epochs} - Training Loss: {training_loss:.6f}, Testing Loss: {testing_loss:.6f}, Training Accuracy: {trainingAcc[-1]:.4f}, Testing Accuracy: {testingAcc[-1]:.4f}")
    
    # Backward
    grad = layers[-1].gradient(Y_train, trainingH)
    for i in range(len(layers) - 2, 0, -1):
        newGrad = layers[i].backward(grad)
        if isinstance(layers[i], FullyConnectedLayer):
            layers[i].updateWeights(grad, learningRate, l2_lambda)
        elif isinstance(layers[i], ConvolutionalLayer):
            layers[i].updateKernels(grad, learningRate, l2_lambda)
        
        grad = newGrad
    
    if len(trainingCrossEntropyLoss) > 2 and np.abs(trainingCrossEntropyLoss[-2] - trainingCrossEntropyLoss[-1]) < tolerance:
        print("Difference in cross entropy loss is below threshold. Terminating early...")
        break

toc = time.perf_counter()
print("Time taken (in seconds):", toc - tic)

print("Epochs:", e + 1)
print("Learning rate:", learningRate)
print("Tolerance:", tolerance)
print()

plt.figure()
plt.plot(trainingCrossEntropyLoss, label='Training')
plt.plot(testingCrossEntropyLoss, label='Testing')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.title(f"LR={learningRate}, # of images={num_images}")
plt.grid()
plt.legend()
plt.savefig("Loss-color-1-kernel.png")
plt.show()

plt.figure()
plt.plot(trainingAcc, label='Training')
plt.plot(testingAcc, label='Testing')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f"LR={learningRate}, # of images={num_images}")
plt.grid()
plt.legend()
plt.savefig("Accuracy-color-1-kernel.png")
plt.show()
