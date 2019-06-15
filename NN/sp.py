import numpy as np
import matplotlib.pyplot as plt
np.random.seed(64)

def step_function(y):
    if y < 0:
        return 0
    else:
        return 1
    
if __name__ == "__main__":
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    w = np.random.rand(3)
    t = np.array([0, 0, 0, 1]) #AND
    #t = np.array([0, 1, 1, 1]) #OR
    #t = np.array([0, 1, 1, 0]) #XOR
    
    alpha = 0.01
    err=[]
    
    for epoch in range(100):
        y_pred=[]
        
        for i in range(4):
            
            #forward network
            out = w[0]*data[i][0] + w[1]*data[i][1]+ w[2]
            y = step_function(out)
            
            #training
            w[0] = w[0] + alpha*((t[i])-y)*data[i][0]
            w[1] = w[1] + alpha*((t[i])-y)*data[i][1]
            w[2] = w[2] + alpha*((t[i])-y)
            
            y_pred.append(y)
            
        #Evaluate
        MSE = 0.5 * np.sum((np.array(y_pred)-t)**2)
        err.append(MSE)
        print('Epoch{}:, Pred:{}, T:{}'.format(epoch, y_pred, t)) 
    
    plt.plot(err)
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.show