import numpy as np

T = 10
A = np.zeros(T)
gamlam = np.array([0.5**i for i in range(T)])
gamlam = np.flip(gamlam)
delta_hat = np.array([1, 5, 10, 15, 20, -10, -15, -20, 0, 100])
delta_hat = np.array([10 for i in range(T)])
# delta_hat = np.array([(-1)**i * 10 for i in range(T)])

for i in range(T):
    A[:i+1] += gamlam[-i-1:] * delta_hat[i]

print(delta_hat)
print(A)