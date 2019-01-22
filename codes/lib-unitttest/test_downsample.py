from aux import downsample
import numpy as np
import matplotlib.pyplot as plt

T_RANGE = [0.0, 1.0]
T = T_RANGE[1] - T_RANGE[0]
DT1 = 0.01 # Times
DT2 = 0.05 # Bins

# Create data
N1 = int(T / DT1) + 1
t1 = np.linspace(T_RANGE[0], T_RANGE[1], N1)
y1 = np.random.normal(0, 1, N1)
for i in range(1, N1):
    y1[i] += y1[i-1]

t2, y2 = downsample(t1, y1, DT2)

plt.figure()
#plt.plot(x, y_much, '.')
plt.plot(t1, y1, 'r.-')
plt.plot(t2, y2, 'g.-')
plt.show()