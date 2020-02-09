"""
@author: Hasnain
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def cost_function(X, y, weights):
    m = len(y)
    predictions = np.dot(X,weights)
    J = np.sum((predictions-y)**2)/(2*m)
    return J

def gradient_descent(X, y, alpha, epochs, weights):
    costs = np.zeros(epochs)
    m = len(y)
    for i in range(epochs):
        hypothesis = np.dot(X,weights)
        error = hypothesis - y
        slope = np.dot(error, X)/m 
        #updating parameters
        weights = weights - alpha*slope
        costs[i] = cost_function(X,y,weights)        
    return weights, costs

if __name__=='__main__':
    df = pd.read_csv("ex1data2.txt", sep=",",header=None)
    df.columns = ["size_ft","bedrooms","price"]
    df['bias'] = 1
    X_ne = np.array(df[['bias','size_ft','bedrooms']].values)
    y_ne = y = np.array(df[['price']].values).flatten()
    
    #feature Normalization
    mean = df['size_ft'].mean()
    stdev = df['size_ft'].std()
    df['size_ft']=(df['size_ft']-mean)/stdev
    df['price']=df['price']/1000
    
    #splitting
    X = np.array(df[['bias','size_ft','bedrooms']].values)
    y = np.array(df[['price']].values).flatten()
    weights = np.array([0,0,0])
    
    alphas = [0.001, 0.003,0.01,0.03,0.1]
    max_iter = 50
    plt.figure()
    iter_ = np.arange(max_iter)
    for lr in alphas:
        (gd_weights, costs) = gradient_descent(X,y,lr,max_iter,weights)
        plt.plot(iter_,costs,'-')    
    plt.legend(alphas, loc='upper right')
    plt.show()
    
    # running gradient descent
    epochs = 1500
    alpha = 0.1
    (gd_weights, costs) = gradient_descent(X,y,alpha,epochs,weights)
    
    # solving with normal equations    
    ne_weights = np.matmul(X_ne.T,X_ne)
    ne_weights = np.linalg.pinv(ne_weights)
    ne_weights = np.matmul(ne_weights,X_ne.T)
    ne_weights = np.matmul(ne_weights,y_ne)
    ne_weights = ne_weights.flatten()
    ### prediction ### 
    
    # with normal equation weights
    ne_pred = ne_weights[0] + ne_weights[1]*1650 + ne_weights[2]*3
    # = 293081.464
    
    # with gradient desscent weights
    x = (1650 - mean)/stdev
    gd_pred = gd_weights[0] + gd_weights[1]*x + gd_weights[2]*3
    gd_pred = gd_pred*1000 
    # = 293023.081
    
    
    x_pred = np.linspace(X[:,1].min(), X[:,1].max(), 30)  
    y_pred = np.linspace(X[:,2].min(), X[:,2].max(), 30)
    xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
    model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T
    model_viz = np.c_[np.ones(len(model_viz)),model_viz] 
    predicted = np.dot(model_viz,gd_weights)
    plt.style.use('default')
    fig = plt.figure(figsize=(10, 3))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    axes = [ax1, ax2]
    for ax in axes:
        ax.plot(X[:,1], X[:,2], y, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
        ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
        ax.set_xlabel('Size of House (Norm)', fontsize=8)
        ax.set_ylabel('Number of bedrooms', fontsize=8)
        ax.set_zlabel('Price of House', fontsize=8)
        ax.locator_params(nbins=4, axis='x')
        ax.locator_params(nbins=5, axis='x')
        
    
    ax1.text2D(0.2, 0.32, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
               transform=ax1.transAxes, color='grey', alpha=0.5)
    
    ax2.text2D(0.3, 0.42, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
               transform=ax2.transAxes, color='grey', alpha=0.5)

    ax1.view_init(elev=28, azim=120)
    ax2.view_init(elev=4, azim=80)
    fig.tight_layout()
        
    
    
    
    
    
    
    
