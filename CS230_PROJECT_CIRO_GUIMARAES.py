# ==================== libraries ==============================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import copy
import math
# =============================================================================


data = np.loadtxt('data2.txt') # loading input data
target = np.loadtxt('target.txt') # loading target data

# ==================== functions ==============================================
def fseries_reconstruct (alpha, omega, time):
    
    w_expanded = omega[:, np.newaxis]                                           # Shape (M,1)
    alpha_expanded = alpha[:, np.newaxis]                                       # Shape (M,1)
    
    return (np.real(np.sum(alpha_expanded * np.exp(1j * w_expanded * time), axis=0)))

def fseries (y):
    M = len(y)
    alpha = np.fft.fftshift(np.fft.fft(y)) / M
    w = 2 * np.pi * np.arange(-M/2, M/2)/23

    return alpha, w


def compute_cost (y, yhat):
    cost = (1/2)*np.sum(y-yhat[:,1])**2 
    cost = np.squeeze(cost)# + np.linalg(pole)
    return (cost)
    

def initialize_parameters(dims):
    parameters = {}
    parameters["RealRes"] = np.random.randn(dims)
    parameters["ImagRes"] = np.random.randn(dims)
    parameters["RealPole"] = np.random.randn(dims)
    parameters["ImagPole"] = np.random.randn(dims)
    return (parameters)

def forward_propagation(X, parameters):
    
    alpha, omega = fseries(X[:,1])
    
    betas = parameters["RealRes"] + 1j * parameters["ImagRes"]               
    mus = parameters["RealPole"] + 1j * parameters["ImagPole"] 
    t = X[:,0]
    
    
    alpha_betas_sum = np.sum(alpha[:, None] / (mus - 1j * omega[:, None]), axis=0)
    betas_mus_sum = np.sum(betas[:, None] / (1j * omega - mus[:, None]), axis=0)

    # Calculate gammas and lambdas
    gammas = betas * alpha_betas_sum
    lambdas = alpha * betas_mus_sum

    # Pre-compute exponentials for efficiency
    omega_exp = np.exp(1j * omega[:, None] * t)
    mus_exp = np.exp(mus[:, None] * t)

    # Calculate output
    u_out = np.real(np.sum(lambdas[:, None] * omega_exp, axis=0) +
                    np.sum(gammas[:, None] * mus_exp, axis=0))
    
    
    caches = (betas, mus, alpha, omega)
    
    return (u_out, caches)


### Unsuccessful vectorization of the gradient of the loss function ###########
# def backward_propagation(X, Yhat, cache):
    
#     betas, mus, alpha, omega = cache
#     parameters = {}
#     parameters["RealRes"] = betas.real
#     parameters["ImagRes"] = betas.imag
#     parameters["RealPole"] = mus.real
#     parameters["ImagPole"] = mus.imag
    
#     Y,_ = forward_propagation(X, parameters)
    
#     grads = {}
#     m = Yhat.shape[0]  # number of samples
#     betas, mus, alpha, omega = cache
#     t = Yhat[:,0]
    
    
#     dbeta = []
#     dmu = []
#     for idx, mu in enumerate(mus):
#         alpha_extended = np.tile(alpha[:, np.newaxis], (1, len(t)))
#         omega_extended = np.tile(omega[:, np.newaxis], (1, len(t)))
#         term1 = alpha_extended/(mu-1j*omega_extended)
#         t_extended = np.tile(t[:, np.newaxis], (1, len(omega))).T
#         term2 = term1*(np.exp(mu*t_extended)-np.exp(1j*omega_extended*t_extended))
#         term3 = np.sum(term2, axis = 0)
#         dbeta.append(np.sum((Y-Yhat[:,1])*term3)/m)
        
#         term4 = betas[idx]*np.sum((alpha_extended/(mu-1j*omega_extended)**2)*(mu-1j*omega_extended-1)*np.exp(omega_extended*t_extended)+np.exp(1j*omega_extended*t_extended), axis = 0)
#         dmu.append(np.sum((Y-Yhat[:,1])*term4)/m)
#     dbeta = np.array(dbeta)
#     dmu = np.array(dmu)

#     # Store real and imaginary parts in grads dictionary
#     grads["dRealRes"] = dbeta.real
#     grads["dImagRes"] = 1j*dbeta.real
#     grads["dRealPole"] = dmu.real
#     grads["dImagPole"] = 1j*dmu.real
    
#     return grads
###############################################################################

def backward_propagation(X, Yhat, cache, epsilon=1):
    
    betas, mus, alpha, omega = cache
    
    parameters = {}
    parameters["RealRes"] = betas.real
    parameters["ImagRes"] = betas.imag
    parameters["RealPole"] = mus.real
    parameters["ImagPole"] = mus.imag
    
    grads = {}
    for key in parameters:
        values = []
        for idx, value in enumerate(parameters[key]):
            params_plus = copy.deepcopy(parameters)
            params_minus = copy.deepcopy(parameters)        
            params_plus[key][idx] += epsilon      
            params_minus[key][idx] -= epsilon
            Yplus,_= forward_propagation(X, params_plus)
            Yminus,_= forward_propagation(X, params_minus)
            loss_plus = compute_cost(Yplus, Yhat)
            loss_minus = compute_cost(Yminus, Yhat)
            values.append((loss_plus - loss_minus) / (2 * epsilon))
        grads['d'+key] = np.array(values)
    return grads
      
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    np.random.seed(seed)  
    m = X.shape[0]
    mini_batches = []
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]
    
    inc = mini_batch_size
    
    num_complete_minibatches = math.floor(m / mini_batch_size)
    
    for k in range (0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k*inc:(k+1)*inc]
        mini_batch_Y = shuffled_Y[k*inc:(k+1)*inc]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size:m]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size:m]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    return (mini_batches)

def initialize_velocity(parameters):


    m = len(parameters["RealRes"])
    v = {}
    
    # Initialize velocity
    v["dRealRes"] = np.zeros(m)
    v["dImagRes"] = np.zeros(m)
    v["dRealPole"] = np.zeros(m)
    v["dImagPole"] = np.zeros(m)
    
    return v

def update_parameters_with_gd(parameters, grads, learning_rate):
 

    parameters["RealRes"] = parameters["RealRes"]-learning_rate* grads["dRealRes"]
    parameters["ImagRes"] = parameters["ImagRes"]-learning_rate* grads["dImagRes"]   
    parameters["RealPole"] = parameters["RealPole"]-learning_rate* grads["dRealPole"]
    parameters["ImagPole"] = parameters["ImagPole"]-learning_rate* grads["dImagPole"] 
    
    return parameters

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    
   
    v["dRealRes"] = beta*v["dRealRes"]+(1-beta)*grads['dRealRes']
    v["dImagRes"] = beta*v["dImagRes"]+(1-beta)*grads['dImagRes']
    v["dRealPole"] = beta*v["dRealPole"]+(1-beta)*grads['dRealPole']
    v["dImagPole"] = beta*v["dImagPole"]+(1-beta)*grads['dImagPole']
    

    parameters["RealRes"] = parameters["RealRes"]-learning_rate* v["dRealRes"]
    parameters["ImagRes"] = parameters["ImagRes"]-learning_rate* v["dImagRes"]   
    parameters["RealPole"] = parameters["RealPole"]-learning_rate* v["dRealPole"]
    parameters["ImagPole"] = parameters["ImagPole"]-learning_rate* v["dImagPole"] 

        
    return parameters, v

def initialize_adam(parameters) :

    m = len(parameters["RealRes"])
    v = {}
    s = {}
    
    v["dRealRes"] = np.zeros(m)
    v["dImagRes"] = np.zeros(m)
    v["dRealPole"] = np.zeros(m)
    v["dImagPole"] = np.zeros(m)
    
    s["dRealRes"] = np.zeros(m)
    s["dImagRes"] = np.zeros(m)
    s["dRealPole"] = np.zeros(m)
    s["dImagPole"] = np.zeros(m)

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):

    
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    

    v["dRealRes"] = beta1*v["dRealRes"]+(1-beta1)*grads["dRealRes"]
    v["dImagRes"] = beta1*v["dImagRes"]+(1-beta1)*grads["dImagRes"]
    v["dRealPole"] = beta1*v["dRealPole"]+(1-beta1)*grads["dRealPole"]
    v["dImagPole"] = beta1*v["dImagPole"]+(1-beta1)*grads["dImagPole"]
    

    v_corrected["dRealRes"] = v["dRealRes"]/(1-beta1**t)
    v_corrected["dImagRes"] = v["dImagRes"]/(1-beta1**t)
    v_corrected["dRealPole"] = v["dRealPole"]/(1-beta1**t)
    v_corrected["dImagPole"] = v["dImagPole"]/(1-beta1**t)
    
    
    
    s["dRealRes"] = beta2*s["dRealRes"]+(1-beta2)*(grads["dRealRes"])**2
    s["dImagRes"] = beta2*s["dImagRes"]+(1-beta2)*(grads["dImagRes"])**2
    s["dRealPole"] = beta2*s["dRealPole"]+(1-beta2)*(grads["dRealPole"])**2
    s["dImagPole"] = beta2*s["dImagPole"]+(1-beta2)*(grads["dImagPole"])**2

    s_corrected["dRealRes"] = s["dRealRes"]/(1-beta2**t)
    s_corrected["dImagRes"] = s["dImagRes"]/(1-beta2**t)
    s_corrected["dRealPole"] = s["dRealPole"]/(1-beta2**t)
    s_corrected["dImagPole"] = s["dImagPole"]/(1-beta2**t)
    
 
    parameters["RealRes"] = parameters["RealRes"] -learning_rate*v_corrected["dRealRes"]/(np.sqrt(s_corrected["dRealRes"])+epsilon)
    parameters["ImagRes"] = parameters["ImagRes"] -learning_rate*v_corrected["dImagRes"]/(np.sqrt(s_corrected["dImagRes"])+epsilon)
    parameters["RealPole"] = parameters["RealPole"] -learning_rate*v_corrected["dRealPole"]/(np.sqrt(s_corrected["dRealPole"])+epsilon)
    parameters["ImagPole"] = parameters["ImagPole"] -learning_rate*v_corrected["dImagPole"]/(np.sqrt(s_corrected["dImagPole"])+epsilon)
    


    return parameters, v, s, v_corrected, s_corrected

def model(X, Y, poleres_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 5000, print_cost = True):

    
    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
    m = X.shape[0]                   # number of training examples
    
    # Initialize parameters
    parameters = initialize_parameters(poleres_dims)
    
    # Initialize the optimizer
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    
    # Optimization loop
    for i in range(num_epochs):
        
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0
        
        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            
            # Forward propagation
            a, caches = forward_propagation(minibatch_X, parameters)

            # Compute cost and add to the cost total
            cost_total += compute_cost(a, minibatch_Y)

            # Backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s, _, _ = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2,  epsilon)
            
        cost_avg = cost_total/m 
        
        # Print the cost every 1000 epoch
        if print_cost and i % 100 == 0:
            print ("Cost after epoch %i: %f" %(i, cost_avg))
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)
        if print_cost and i % 200 == 0:
            learning_rate = learning_rate/100
                
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters
# =============================================================================


# ====================== Signal Parameters ====================================
Qref = 2000
Nf = 256*2
poleres_dims = 20   
T = data[-1,0]                                                             
ts = np.linspace(0, T, Nf, endpoint=False)                                     

# X: input
interp_func_f = interp1d(data[:,0], data[:,2])
f = interp_func_f(ts)/Qref

f = np.ones(len(ts))*Qref
X = np.concatenate((ts[:, np.newaxis], f[:, np.newaxis]), axis=1)

# Yhat: target output
interp_func_yhat = interp1d(target[:,0], target[:,2])
yhat = interp_func_yhat(ts)
Yhat = np.concatenate((ts[:, np.newaxis], yhat[:, np.newaxis]), axis=1)


# =============================================================================


# ================== Source-term ==============================================
alpha, w = fseries(X)

beta = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
lr = 0.1


parameters = model(X, Yhat, poleres_dims, optimizer = "adam" , learning_rate = lr, mini_batch_size = 64, beta = beta,
          beta1 = beta1, beta2 = beta2,  epsilon = epsilon, num_epochs = 1000, print_cost = True)


Y,cache = forward_propagation(X, parameters)


# =============================================================================


