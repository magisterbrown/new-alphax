import time
import random
import numpy as np
els = 5
arrs = np.random.uniform(-1,1,els)
print(arrs)
reals = np.random.randint(-1,2,els)
print(reals)
var = np.var(reals-arrs)
print(var)
div = var/np.var(reals)
print(div)
print(1-div)
