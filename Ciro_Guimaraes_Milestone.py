# ==================== libraries ==============================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# =============================================================================



# =============================================================================
# This is the implementation of an optimization of an arbitrary set of poles
# and residues to fit a given solution 'u'. The Green's function here is 
# known, but we will attempt to predict it from data alone. 
# =============================================================================



# ==================== functions ==============================================
def fseries (y, t):
    # Args: y: input signal.
    # Returns: coefficients and frequencies of the Fourier series stacked.
    
    M = len(y)
    alpha = np.fft.fftshift(np.fft.fft(y)) / M
    w = 2 * np.pi * np.arange(-M/2, M/2)

    w_expanded = w[:, np.newaxis]                                               # Shape (M,1)
    alpha_expanded = alpha[:, np.newaxis]                                       # Shape (M,1)
    
    y_fseries = np.real(np.sum(alpha_expanded * np.exp(1j * w_expanded * t), axis=0))  # Shape (M,)
    return y_fseries, alpha, w

def LNO(alpha, omega, betas, mus, ts):
    # Args: alpha: coefficients.
    # omega: frequencies.
    # betas: residues.
    # mus: poles.
    # ts: time vector.
    # Returns: coefficients and frequencies of the Fourier series stacked.
    
    gammas = betas * np.sum(alpha[:, None] / (mus - 1j * omega[:, None]), axis=0)

    lambdas = alpha * np.sum(betas[:,None] / (1j * omega - mus[:, None]), axis=0)

    u_out = (
        np.real(np.sum(lambdas[:, None] * np.exp(1j * omega[:, None] * ts), axis=0)) +
        np.sum(gammas[:, None] * np.exp(mus[:, None] * ts), axis=0)
    )
    return u_out

def loss (params):
    m = len(params) // 4  # Each beta and mu has real and imaginary parts
    residues = params[:m] + 1j * params[m:2*m]  # First half for betas
    poles = params[2*m:3*m] + 1j * params[3*m:]  # Second half for mus
    y = LNO(alpha,w,residues,poles, ts)
    
    return (np.linalg.norm(y-convolution[:Nf]))
# =============================================================================


# ====================== Signal Parameters ====================================
Nf = 256                                                                        # Number of sample points
T = 1.0                                                                         # Time period
ts = np.linspace(0, T, Nf, endpoint=False)                                      # Time vector
# =============================================================================


# ================== Source-term ==============================================
q_schedule = np.array([1,0,4,0,0,-4,0])                                         # Flowrate History


f = np.zeros(Nf)
for i in range (len(q_schedule)-1):
    f[i*int(Nf/len(q_schedule)):(i+1)*int(Nf/len(q_schedule))] = q_schedule[i]
f[(len(q_schedule)-1)*int(Nf/len(q_schedule)):] = q_schedule[-1]

f_new = np.ones(Nf)
ts_new = ts = np.linspace(0, T, Nf, endpoint=False)  

f_manual_reconstructed, alpha, w = fseries(f, ts)                               # Reconstructed using Fourier Series
f_manual_reconstructed_new, alpha_new, w_new = fseries(f_new, ts_new)
# =============================================================================


# ====================== Green's Function =====================================
# This section initializes the Green's function using the Pole and Residule method

Nl = 2                                                                          # Number of poles/residues
betas = np.random.randn(Nl) + 1j * np.random.randn(Nl)                          # Initialization of residues
mus = np.random.randn(Nl) + 1j * np.random.randn(Nl)                            # Initialization of poles

# =============================================================================


# ================================== Main =====================================
g = np.sin(3*np.pi * ts)                                                        # Known Green's Function

convolution = np.convolve(g, f, mode='full')/Nf                                 # Convolution using Numpy (ground truth)

initial_guess = np.concatenate((betas.real, betas.imag, mus.real, mus.imag,))

result = minimize(loss, initial_guess, method='L-BFGS-B', options={'maxiter': 10000, 'gtol': 1e-5})

optimized_betas = result.x[:Nl] + 1j * result.x[Nl:2*Nl]
optimized_mus = result.x[2*Nl:3*Nl] + 1j * result.x[3*Nl:]  

u0 = LNO(alpha, w, betas, mus, ts)
u = LNO(alpha,w,optimized_betas,optimized_mus, ts)

unew = LNO(alpha_new,w_new,optimized_betas,optimized_mus, ts_new)


# =============================================================================



# ================== Post-processing ==========================================
print('==================== Poles =================')
for i in range(len(optimized_mus)):
    print(f"μ{i}: {optimized_mus[i]:.3f}")
print('=============================================\n')

print('================== Residues ==================')
for i in range(len(optimized_betas)):
    print(f"β{i}: {optimized_betas[i]:.3f}")
print('==============================================')


conv_resample = convolution[:Nf:(int(Nf/60))]

fig = plt.figure(figsize=(14, 9), facecolor='lightgrey')
gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])  

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])


ax1.plot(ts, u, lw=4, color='black', label='LNO - Optimized Poles/Residues',zorder = 1)
ax1.plot(ts, u0, lw=4, color='grey', label='Initial Guess',zorder = 10)
ax1.scatter(np.linspace(0, 1, len(conv_resample), endpoint=False), conv_resample, 
            s = 60, color='red',edgecolor='black', label='Numpy Convolution', 
            zorder = 100)
ax1.set_xlim(-0.05,1.05)
ax1.set_ylabel('Convolution f*G', fontsize = 14, labelpad=20)
ax1.grid()
ax1.legend(loc='upper left', fontsize='large')


ax1.label_outer()  

ax2.plot(ts,f, label='Source Term (original)', color='black', lw=6, zorder=0)
ax2.plot(ts,f_manual_reconstructed, label='Source Term (fseries)', color='red', 
         lw=3, linestyle='--', zorder=1)
ax2.set_xlabel('t (samples)', fontsize = 14)  
ax2.set_ylabel('f(t)', fontsize = 14, labelpad=20)
ax2.legend(loc='upper right')
ax2.axhline(y=0, color='orange', linestyle='--', linewidth=1.5)
ax2.grid()
ax2.set_xlim(-0.05,1.05)
ax2.fill_between(ts, f, 0, where=(f > 0), color='olive', alpha=0.5, zorder=-1)
ax2.fill_between(ts, f, 0, where=(f < 0), color='lightblue', alpha=0.5, zorder=-1)



plt.suptitle('Convolution: LNO vs. Conventional Approach',fontweight='bold', 
             fontsize = 18, y=0.95)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.05)  
plt.show()
