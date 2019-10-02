import sys
import numpy as np
from scipy.stats import pearsonr
from matplotlib import pyplot as plt

def parse_result(result):

	results = np.load(result)

	means = np.mean(results,axis = 0)
	stds = np.std(results,axis = 0)

	print("Treatment group,Model CC,Baseline CC")
	for i in range(means.shape[0]):
	    print(str(int(means[i,0])) + ", " + str(float('%.2g' % means[i,1])) + " (" + str(float('%.2g' % stds[i,1])) + "), " + str(float('%.2g' % means[i,2])))

	return np.sum(means[:,1])

def parse_profile(data,plot=False):

	data = np.load(data)
	ae_profiles = data[0]
	control_profiles = data[1]
	target_profiles = data[2]

	for i in range(ae_profiles.shape[1]):
		print(pearsonr(ae_profiles[:,i],target_profiles[:,i])[0], pearsonr(control_profiles[:,i],target_profiles[:,i])[0])

	ae_ccs = []
	control_ccs = []
	count = 0

	for i in range(ae_profiles.shape[0]):
		ae_ccs.append(pearsonr(ae_profiles[i,:],target_profiles[i,:])[0])
		control_ccs.append(pearsonr(control_profiles[i,:],target_profiles[i,:])[0])
		if ae_ccs[-1] > control_ccs[-1]:
			count += 1

	print(count / i)

	if plot:
		plt.title('Correlation across treatments')
		plt.hist(ae_ccs,20,color='b',alpha=0.5)
		plt.hist(control_ccs,20,color='g',alpha=0.5)
		plt.show()

	return np.median(ae_ccs)

