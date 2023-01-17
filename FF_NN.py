# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:42:20 2023

@author: Daniel Matthew

1. Takes inputs as a matrix (2D array of numbers)
2. Multiplies the input by a set of weights
3. Applies an activation function
4. Returns an output (a prediction)
5. Error is calculated by taking the difference from the desired output from the
data and the predicted output. This creates our gradient descent, which we
can use to alter the weights.
6. The weights are then altered slightly according to the error.
7. To train, this process is repeated 1,000+ times. The more the data is trained
upon, the more accurate our outputs will be.



"""

import numpy as np

# X = (hours studying, hours sleeping), y = score on test
x_all = np.array(([2, 9], [1, 5], [3, 6], [5, 10]), dtype=float) # input data

y = np.array(([92], [86], [89]), dtype=float) # output

x_all = x_all/np.max(x_all, axis=0) # scaling input data

y = y/100 # scaling output data (max test score is 100)

# split data
X = np.split(x_all, [3])[0] # training data
x_predicted = np.split(x_all, [3])[1] # testing data

class neural_network(object):
    def __init__(self):
        #parameters
     self.inputSize = 2
     self.outputSize = 1
     self.hiddenSize = 3
     #weights
     #first, our weight is a 2x3 matrix from input to hidden layer
     self.W1 = np.random.randn(self.inputSize,self.hiddenSize)
     #hen it is 3x1 from hidden to output layer 
     self.W2 = np.random.randn(self.hiddenSize,self.outputSize) 

#define activation function
# this function gives us value close to 1 if possitive, close to 0 if neg
# and between 0-1 if input close to 0
    def sigmoid(self, s):
        return 1/(1+np.exp(-s))


#define forward prop
    def forward(self, X):
    #forward prop through network
        self.z = np.dot(X, self.W1) #get dot prod of Input, and 2x3 weights maxtrix
        self.z2  = self.sigmoid(self.z) #pass activation function to the dot producxt 
    
        #answer
        # dot product of hidden layer (z2) and set of 3x1 weights (w2) 
        self.z3 = np.dot(self.z2, self.W2)
        #final activation func, final ans 
        o = self.sigmoid(self.z3)
        return o; 
    
    def sigmoidPrime(self, s): 
        return s * (1 - s)

    #now define back prop
    def backward(self, X, y, o): 
        self.o_error = y - o #error in output
        
        #derivative of sigmoid multiplied by OUTPUT error
        self.o_delta =self.o_error * self.sigmoidPrime(o)
        
        # how much hidden latyer contributed to error? 
        self.z2_error = self.o_delta.dot(self.W2.T)
        
        #deriv of sigmoid to z2 error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) 
        
        #adjust werights from first set, input to hidden layer weight
        self.W1 += X.T.dot(self.z2_delta)
        
        #adjust second set, hidden to output layer
        self.W2 =+ self.z2.T.dot(self.o_delta)
        
        
    # now we train the netwrork with both forward and back
    # lets make a function for this so we can do it many times for good training 
    def train(self, X, y): 
        o = self.forward(X)
        self.backward(X, y, o)

#now we can define our neural network
nn = neural_network() 

#define our forward prop output
o = nn.forward(X)

print("predicted: " + str(o))
print("actual: " + str(y))


"""

print("predicted: " + str(o))
predicted: [[0.75088084]
 [0.74776836]
 [0.73305462]]

print("actual: " + str(y))
actual: [[0.92]
 [0.86]
 [0.89]]

results are very off. How do we fix it? back propogation. lets implement it 

first we must calculate error
we can do this by going predicted - actual squared, /2 
we do this for all our points, this is known as MSE 

Now that we have a loss function, lets use calculus to find our min loss 

to find min, sub f(x) with the equation 
f(x + h) - f(x) 
/ h

so, if f(x) = x^2 - 2x 

we do f'(x) = ((x+h)^2 - 2(x+h)) - (x^2 - 2x ) 
/ h


expand to 
f'(x) = x^2 + 2xh + h^2 -2x - 2h -x^2 + 2x  
/h 

simplify to 
 f'(x) = x^2 - X^2 + 2xh + h^2  -2x + 2x -2h 
 /h 
 
 f'*(x) = 2xh + h^2 - 2h / h

 lets factor out h 
 f'*(x) = h(2x + h -2 ) 
 / h
 
 cancel out our h 
 f'*(x) = 2x + h - 2
 
 get rid of h 
 
 f'*(x) = 2x - 2 
 this is the derivative function which is slope at each point
 now we can find slope = 0 to find min and max 
 
 
 0 = 2x - 2
 2 = 2x 
 2/2 = x
 x = 1 
 
 What gradient descent does is helps us find the derivatives at some of the 
 given weights. This helps us find what descreases our MSE derivative 
 WE musst find the derivative of our sigmoid function 
 this is given as 
""" 


    
    
  #lets define loss
nn = neural_network()
for i in range(1000): # trains the nn 1,000 times
 print("Input: \n" + str(X))
 print("Actual Output: \n" + str(y))
 print("Predicted Output: \n" + str(nn.forward(X)))
# print("Loss: \n" + ((o - y)**2) /2) # mean squared error
 print("\n")
 nn.train(X, y)
 print("Loss: \n" + str(np.mean(np.square(y - nn.forward(X))))) # mean squared error





nn.forward(x_predicted)










