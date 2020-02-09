"""
@author: Hasnain
"""
import pandas as pd
import matplotlib.pylab as plt
import numpy as np

"""
Vectorized Implementation
"""
def cost_function(X, y, thetas):
    m = len(y)
    predictions = np.dot(X,thetas)
    J = np.sum((predictions-y)**2)/(2*m)
    return J

def gradient_descent(X, y, alpha, epochs, thetas):
    costs = np.zeros(epochs)
    m = len(y)
    for i in range(epochs):
        hypothesis = np.dot(X,thetas)
        error = hypothesis - y
        slope = np.dot(error, X)/m 
        #updating parameters
        thetas = thetas - alpha*slope
        costs[i] = cost_function(X,y,thetas)        
    return thetas, costs
    
if __name__=='__main__':
    epochs = 500
    alpha = 0.01
    df = pd.read_csv("ex1data1.txt", sep=",",header=None)
    df.columns = ["population","profit"]
    df.plot(kind='scatter',x="population",y="profit")
    df['bias'] = 1
    X = np.array(df[['bias','population']].values)
    y = np.array(df[['profit']].values).flatten()
    thetas = np.array([0,0])
    #running gradient descent
    (new_thetas, costs) = gradient_descent(X,y,alpha,epochs,thetas)
    # Plotting the best fit line
    lineX = np.linspace(0, 25, 20)
    lineY = [new_thetas[0] + new_thetas[1]*x for x in lineX]
    plt.figure(figsize=(10,6))
    plt.plot(df[['population']].values, df[['profit']].values, 'x')
    plt.plot(lineX, lineY, '-')
    plt.axis([0,25,-5,25])
    plt.xlabel('population')
    plt.ylabel('profit')
    plt.title('Profit vs Population with Linear')
