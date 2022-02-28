#########################################
## Find optimal k for Kmeans clustering
#########################################
'''
[Order of function implementation]
1) main_sse
'''
import os
from os import listdir
from os.path import isfile, join, exists
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from kneed import KneeLocator
'''
[Definition]
- SSE: sum of squared distances of samples to their closest cluster center (Inertia)
- The elbow was defined as the point where the derivatives changes the most
'''

BASE_DIR = 'X:/path/myfolder'
DATA_DIR = BASE_DIR + '/data'

def set_subjdir(subID): return f'{DATA_DIR}/{subID}'



'''
[kmeans_sse]
To perform K-means clustering when 1<K<=10 and compute SSE 

Input:  1) {subjdir}/tracto/fs_default.conn.matrix.npy
	2) {subjdir}/tracto/fs_default.seed_idx.npy
Output: {subjdir}/cluster/{hemi}-sse.npy
'''
def kmeans_sse(subID, hemi):
	subjdir = set_subjdir(subID)
	conn_dir = join(subjdir, 'tracto')
	clust_dir = join(subjdir, 'cluster')
	out_dir = f'{clust_dir}/optimal_k'
	if not os.path.exists(clust_dir):
		os.makedirs(clust_dir)

	# 1) Load input files (connectivity matrix, seed indices)
	# the connectivity map include null seed & null target voxels at first row & column
	conn_mat = np.load(join(conn_dir, 'fs_default.conn.matrix.npy'))[1:, 1:]
	seed_idx = np.load(join(conn_dir, 'fs_default.seed_idx.npy'))
	
	# 2) Consider voxels with connectivity value larger than 100
	valid_seed = np.where(conn_mat.sum(axis=1) >= 100)
	valid_conn = conn_mat[valid_seed]
	valid_conn = valid_conn / valid_conn.sum(axis=1).reshape(-1,1).astype(np.float64)
	valid_idx = np.transpose(seed_idx)[valid_seed]

	# 3) Divide data into Left/Right
	div_x = np.argmax(np.diff(valid_idx[:,0])) + 1	# the start point of x-axis in left hemisphere 

	n_vox = len(valid_idx)
	hemi_set = np.array(div_x*['R'] + (n_vox-div_x)*['L'])

	# Remove target ROIs with low connectivity strength
	if hemi == 'L':
		target_roi = np.arange(42)
		ban_roi = np.array([5, 13, 25, 31, 32, 33, 42]) - 1
	elif hemi == 'R':
		target_roi = np.arange(42,84)
		ban_roi = np.array([49, 54, 52, 74, 80, 81, 82]) - 1

	valid_roi = np.setdiff1d(target_roi, ban_roi, assume_unique=True)
	hemi_conn = valid_conn[hemi_set==hemi][:,valid_roi]
	hemi_idx = valid_idx[hemi_set==hemi]

	# 4) Perform K-means clustering changing K and compute SSE
	sse = np.zeros(9)
	for ncluster in range(2, 11):
		clf = KMeans(n_clusters=ncluster, random_state=0).fit(hemi_conn)
		orig_labels = clf.predict(hemi_conn)
		centroids = clf.cluster_centers_
		curr_sse = 0
		for i in range(len(orig_labels)):
			curr_center = centroids[orig_labels[i]]
			curr_sse += np.sum((hemi_conn[i,:] - curr_center)**2)
		sse[ncluster-2] = curr_sse
	np.save(f'{out_dir}/{hemi}-sse.npy', sse)
	return sse


'''
[main_sse]
To perform K-means clustering for whole subjects, compute averaged SSE, and plot it 
- kmeans_sse

Output: {BASE_DIR}/clustFC/{hemi}-sse.npy
'''
def main_sse(a=0, b=0):
	sublist = sorted(listdir(DATA_DIR))
	if type(a) == str:
		a = sublist.index(a)
	
	if b==0:
		sublist = sublist[a:]
	else:
		sublist = sublist[a:b]  # 162026 ~ 793465

	for hemi in ['L', 'R']:
		sse = np.zeros((len(sublist), 9))
		for sidx, subID in enumerate(sublist):
			print(f'{sidx+1}th sub - {subID} - clustering\n')
			sse[sidx] = kmeans_sse(subID, hemi)
		np.save(f'{BASE_DIR}/clustFC/{hemi}-sse.npy', sse)

		# Plot SSE and find elbow
		x = np.arange(2,11)
		avg = np.mean(sse, axis=0)
		kn = KneeLocator(x, avg, curve='convex', direction='decreasing')
		plt.plot(x, avg)
		plt.xlabel('k')
		plt.ylabel('SSE')
		plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
		plt.savefig(f'{BASE_DIR}/clustFC/figure/{hemi}-sse.png')
		plt.close()
