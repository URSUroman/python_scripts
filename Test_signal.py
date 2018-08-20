
#from third-party
import numpy as np
from scipy.io import loadmat
import scipy.signal
import scipy as sp
import matplotlib.pyplot as plt
import os
import glob


npdir = os.path.dirname(np.__file__)
print("NumPy is installed in %s" % npdir)

spdir = os.path.dirname(sp.__file__)
print("SciPy is installed in %s" % spdir)


print(np.__version__)
#
print(sp.__version__)

padlen = 5
a = np.zeros(512)
a[0] = 1  # make an "all-zero filter"
b = np.ones(512)
rawsong = np.ones(100000)


## Begin of Test
#print(np.__version__)
#
#print(sp.__version__)
#
##np.test()
##sp.test()
#
#np.show_config()
#
## End of Test


print('filtfilt bbegin')
filtsong = scipy.signal.filtfilt(b, a, rawsong, padlen=padlen)
print('filtfilt done')

	
	

	
