import numpy as np

x = np.arange(6).reshape(2,3)

# np.argwhere(x>1)
print(np.argwhere(x>1))
print(np.argwhere(x>1).flatten())