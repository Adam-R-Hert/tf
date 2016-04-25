import numpy as np
data = np.genfromtxt('input.csv', delimiter=',')

data = data.T

print(data.shape)
print(data[1][298])

exit

