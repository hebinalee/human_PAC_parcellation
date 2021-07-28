#########################################
## Find optimal k for Kmeans clustering
#########################################

import os
from os import listdir
from os.path import isfile, join, exists
import numpy as np
import matplotlib.pyplot as plt
if os.name == 'nt':
	store4 = 'X:/'
else:
	store4 = '/store4/'
import nibabel as nib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from kneed import KneeLocator


def find_elbow(coeff):
	x = np.arange(2,11)
	data = np.transpose(np.vstack((x, coeff)))
	theta = np.arctan2(data[:, 1].max() - data[:, 1].min(), data[:, 0].max() - data[:, 0].min())
	# make rotation matrix
	co = np.cos(theta)
	si = np.sin(theta)
	rotation_matrix = np.array(((co, -si), (si, co)))
	# rotate data vector
	rotated_vector = np.dot(rotation_matrix, np.transpose(data))
	#rotated_vector = np.array([[co*data[i,0]-si*data[i,1], si*data[i,0]+co*data[i,1]] for i in range(9)])
	# return index of elbow
	return x[np.where(rotated_vector[1,:] == rotated_vector[1,:].min())[0][0]]

def kmeans_silhouette(subID, hemi):
	subpath = join(store4, 'hblee/4.MPI/4.clustFC/data/', subID)
	path_conn = join(subpath, 'tracto')
	path_clust = join(subpath, 'cluster4')
	if not os.path.exists(path_clust):
		os.makedirs(path_clust)

	ref_anat = nib.load(join(subpath, 'T1w/T1w_acpc_dc_restore_brain.nii.gz'))
	# the connectivity map include null seed & null target voxels at first row & column
	conn_mat = np.load(join(path_conn, 'fs_default.conn.matrix.npy'))[1:, 1:]
	seed_idx = np.load(join(path_conn, 'fs_default.seed_idx.npy'))
	
	valid_seed = np.where(conn_mat.sum(axis=1) >= 100)
	valid_conn = conn_mat[valid_seed]
	#valid_conn = np.where(conn_mat[valid_seed] > 0, 1,0) + 0.1*np.log10(10*conn_mat[valid_seed]+1)
	valid_conn = valid_conn / valid_conn.sum(axis=1).reshape(-1,1).astype(np.float64)
	valid_idx = np.transpose(seed_idx)[valid_seed]

	# div_x is the start point of x-axis in left hemisphere
	div_x = np.argmax(np.diff(valid_idx[:,0])) + 1

	n_vox = len(valid_idx)
	hemi_set = np.array(div_x*['R'] + (n_vox-div_x)*['L'])

	#ban_roi = np.array([29, 33, 34, 78, 82, 83]) - 1  # + [5,13,25,31,32,42,49,54,62,74,80,81]
	if hemi == 'L':
		target_roi = np.arange(42)
		#ban_roi = np.array([5, 13, 25, 29, 31, 32, 33, 34, 42]) - 1
		ban_roi = np.array([5, 13, 25, 31, 32, 33, 42]) - 1
	elif hemi == 'R':
		target_roi = np.arange(42,84)
		#ban_roi = np.array([49, 54, 52, 74, 78, 80, 81, 82, 83]) - 1
		ban_roi = np.array([49, 54, 52, 74, 80, 81, 82]) - 1

	valid_roi = np.setdiff1d(target_roi, ban_roi, assume_unique=True)
	hemi_conn = valid_conn[hemi_set==hemi][:,valid_roi]
	hemi_idx = valid_idx[hemi_set==hemi]

	silhouette = np.zeros(9)
	for ncluster in [2,3,4,5,6,7,8,9,10]:
		clf = KMeans(n_clusters=ncluster, random_state=0).fit(hemi_conn)
		orig_labels = clf.predict(hemi_conn)
		silhouette[ncluster-2] = silhouette_score(hemi_conn, orig_labels)
	np.save(path_clust + '/%s-silhouette.npy' %hemi, silhouette)
	return silhouette


def kmeans_elbow(subID, hemi):
	subpath = join(store4, 'hblee/4.MPI/4.clustFC/data/', subID)
	path_conn = join(subpath, 'tracto')
	path_clust = join(subpath, 'cluster4')
	if not os.path.exists(path_clust):
		os.makedirs(path_clust)

	ref_anat = nib.load(join(subpath, 'T1w/T1w_acpc_dc_restore_brain.nii.gz'))
	# the connectivity map include null seed & null target voxels at first row & column
	conn_mat = np.load(join(path_conn, 'fs_default.conn.matrix.npy'))[1:, 1:]
	seed_idx = np.load(join(path_conn, 'fs_default.seed_idx.npy'))
	
	valid_seed = np.where(conn_mat.sum(axis=1) >= 100)
	valid_conn = conn_mat[valid_seed]
	#valid_conn = np.where(conn_mat[valid_seed] > 0, 1,0) + 0.1*np.log10(10*conn_mat[valid_seed]+1)
	valid_conn = valid_conn / valid_conn.sum(axis=1).reshape(-1,1).astype(np.float64)
	valid_idx = np.transpose(seed_idx)[valid_seed]

	# div_x is the start point of x-axis in left hemisphere
	div_x = np.argmax(np.diff(valid_idx[:,0])) + 1

	n_vox = len(valid_idx)
	hemi_set = np.array(div_x*['R'] + (n_vox-div_x)*['L'])

	#ban_roi = np.array([29, 33, 34, 78, 82, 83]) - 1  # + [5,13,25,31,32,42,49,54,62,74,80,81]
	if hemi == 'L':
		target_roi = np.arange(42)
		#ban_roi = np.array([5, 13, 25, 29, 31, 32, 33, 34, 42]) - 1
		ban_roi = np.array([5, 13, 25, 31, 32, 33, 42]) - 1
	elif hemi == 'R':
		target_roi = np.arange(42,84)
		#ban_roi = np.array([49, 54, 52, 74, 78, 80, 81, 82, 83]) - 1
		ban_roi = np.array([49, 54, 52, 74, 80, 81, 82]) - 1

	valid_roi = np.setdiff1d(target_roi, ban_roi, assume_unique=True)
	hemi_conn = valid_conn[hemi_set==hemi][:,valid_roi]
	hemi_idx = valid_idx[hemi_set==hemi]

	distortions = np.zeros(9)
	for ncluster in [2,3,4,5,6,7,8,9,10]:
		clf = KMeans(n_clusters=ncluster, random_state=0).fit(hemi_conn)
		orig_labels = clf.predict(hemi_conn)
		distortions[ncluster-2] = sum(np.min(cdist(hemi_conn, clf.cluster_centers_, 'euclidean'),axis=1)) / hemi_conn.shape[0]
		#distortions[ncluster-2] = clf.inertia_
	np.save(path_clust + '/%s-distortion.npy' %hemi, distortions)
	return distortions


def kmeans_sse(subID, hemi):
	subpath = join(store4, 'hblee/4.MPI/4.clustFC/data/', subID)
	path_conn = join(subpath, 'tracto')
	path_clust = join(subpath, 'cluster4')
	if not os.path.exists(path_clust):
		os.makedirs(path_clust)

	ref_anat = nib.load(join(subpath, 'T1w/T1w_acpc_dc_restore_brain.nii.gz'))
	# the connectivity map include null seed & null target voxels at first row & column
	conn_mat = np.load(join(path_conn, 'fs_default.conn.matrix.npy'))[1:, 1:]
	seed_idx = np.load(join(path_conn, 'fs_default.seed_idx.npy'))
	
	valid_seed = np.where(conn_mat.sum(axis=1) >= 100)
	valid_conn = conn_mat[valid_seed]
	#valid_conn = np.where(conn_mat[valid_seed] > 0, 1,0) + 0.1*np.log10(10*conn_mat[valid_seed]+1)
	valid_conn = valid_conn / valid_conn.sum(axis=1).reshape(-1,1).astype(np.float64)
	valid_idx = np.transpose(seed_idx)[valid_seed]

	# div_x is the start point of x-axis in left hemisphere
	div_x = np.argmax(np.diff(valid_idx[:,0])) + 1

	n_vox = len(valid_idx)
	hemi_set = np.array(div_x*['R'] + (n_vox-div_x)*['L'])

	#ban_roi = np.array([29, 33, 34, 78, 82, 83]) - 1  # + [5,13,25,31,32,42,49,54,62,74,80,81]
	if hemi == 'L':
		target_roi = np.arange(42)
		#ban_roi = np.array([5, 13, 25, 29, 31, 32, 33, 34, 42]) - 1
		ban_roi = np.array([5, 13, 25, 31, 32, 33, 42]) - 1
	elif hemi == 'R':
		target_roi = np.arange(42,84)
		#ban_roi = np.array([49, 54, 52, 74, 78, 80, 81, 82, 83]) - 1
		ban_roi = np.array([49, 54, 52, 74, 80, 81, 82]) - 1

	valid_roi = np.setdiff1d(target_roi, ban_roi, assume_unique=True)
	hemi_conn = valid_conn[hemi_set==hemi][:,valid_roi]
	hemi_idx = valid_idx[hemi_set==hemi]

	sse = np.zeros(9)
	for ncluster in [2,3,4,5,6,7,8,9,10]:
		clf = KMeans(n_clusters=ncluster, random_state=0).fit(hemi_conn)
		orig_labels = clf.predict(hemi_conn)
		centroids = clf.cluster_centers_
		curr_sse = 0
		for i in range(len(orig_labels)):
			curr_center = centroids[orig_labels[i]]
			curr_sse += np.sum((hemi_conn[i,:] - curr_center)**2)
		sse[ncluster-2] = curr_sse
	np.save(path_clust + '/%s-sse.npy' %hemi, sse)
	return sse


def main_silhouette(a, b):
	datapath = store4 + 'hblee/4.MPI/4.clustFC/data/'
	sublist = listdir(datapath)
	#filelist = [join(datapath, x, 'T1w') for x in listdir(datapath) if len(x)==6]
	if type(a) == str:
		a = sublist.index(a)
	
	if b==0:
		sublist = sublist[a:]
	else:
		sublist = sublist[a:b]  # 162026 ~ 793465

	for hemi in ['L', 'R']:
		silhouette = np.zeros((len(sublist), 9))
		for sidx, sname in enumerate(sublist):
			print('%dth sub - %s - clustering\n' %(sidx+1, sname))
			silhouette[sidx] = kmeans_silhouette(sname, hemi)
		np.save(store4 + 'hblee/4.MPI/4.clustFC/%s-silhouette.npy' %hemi, silhouette)


def main_elbow(a, b):
	datapath = store4 + 'hblee/4.MPI/4.clustFC/data/'
	sublist = listdir(datapath)
	#filelist = [join(datapath, x, 'T1w') for x in listdir(datapath) if len(x)==6]
	if type(a) == str:
		a = sublist.index(a)
	
	if b==0:
		sublist = sublist[a:]
	else:
		sublist = sublist[a:b]  # 162026 ~ 793465

	for hemi in ['L', 'R']:
		distortions = np.zeros((len(sublist), 9))
		for sidx, sname in enumerate(sublist):
			print('%dth sub - %s - clustering\n' %(sidx+1, sname))
			distortions[sidx] = kmeans_elbow(sname, hemi)
		np.save(store4 + 'hblee/4.MPI/4.clustFC/%s-distortion.npy' %hemi, distortions)


def main_sse(a, b):
	datapath = store4 + 'hblee/4.MPI/4.clustFC/data/'
	sublist = listdir(datapath)
	#filelist = [join(datapath, x, 'T1w') for x in listdir(datapath) if len(x)==6]
	if type(a) == str:
		a = sublist.index(a)
	
	if b==0:
		sublist = sublist[a:]
	else:
		sublist = sublist[a:b]  # 162026 ~ 793465

	for hemi in ['L', 'R']:
		sse = np.zeros((len(sublist), 9))
		for sidx, sname in enumerate(sublist):
			print('%dth sub - %s - clustering\n' %(sidx+1, sname))
			sse[sidx] = kmeans_sse(sname, hemi)
		np.save(store4 + 'hblee/4.MPI/4.clustFC/%s-sse.npy' %hemi, sse)

		avg = np.mean(sse, axis=0)
		plt.plot(np.arange(2,11), avg)
		plt.xlabel('k')
		plt.ylabel('SSE')
		plt.savefig(store4 + 'hblee/4.MPI/4.clustFC/figure/%s-sse.png' %hemi)
		plt.close()


def saveall(hemi):
	datapath = store4 + 'hblee/4.MPI/4.clustFC/data/'
	sublist = listdir(datapath)

	silhouette = np.zeros((len(sublist), 9))
	for sidx, sname in enumerate(sublist):
		print('%dth sub - %s - clustering\n' %(sidx+1, sname))
		filepath = join(store4, 'hblee/4.MPI/4.clustFC/data/', sname, 'cluster4', 'R-inertia.npy')
		silhouette[sidx] = np.load(filepath)
	np.save(store4 + 'hblee/4.MPI/4.clustFC/R-inertia.npy', silhouette)


def rename(hemi):
	datapath = store4 + 'hblee/4.MPI/4.clustFC/data/'
	sublist = listdir(datapath)

	for sidx, sname in enumerate(sublist):
		print('%dth sub - %s - clustering\n' %(sidx+1, sname))
		filepath = join(store4, 'hblee/4.MPI/4.clustFC/data/', sname, 'cluster4/')
		os.rename(filepath+'%s-elbow.npy' %hemi, filepath + '%s-inertia.npy' %hemi)


def savefig(hemi):
	#silhouette = np.load(store4 + 'hblee/4.MPI/4.clustFC/%s-silhouette.npy' %hemi)
	#avg = np.mean(silhouette, axis=0)
	#plt.plot(np.arange(2,11), avg)
	#plt.xlabel('k')
	#plt.ylabel('silhouette coefficient')
	#plt.savefig(store4 + 'hblee/4.MPI/4.clustFC/%s-silhouette.png' %hemi)
	#plt.close()

	x = np.arange(2,11)
	elbow = np.load(store4 + 'hblee/4.MPI/4.clustFC/%s-inertia.npy' %hemi)
	avg = np.mean(elbow, axis=0)
	kn = KneeLocator(x, avg, curve='convex', direction='decreasing')
	plt.plot(x, avg)
	plt.xlabel('k')
	plt.ylabel('Inertia')
	plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
	plt.savefig(store4 + 'hblee/4.MPI/4.clustFC/figure/%s-inertia.png' %hemi)
	plt.close()

	elbow = np.load(store4 + 'hblee/4.MPI/4.clustFC/%s-distortion.npy' %hemi)
	avg = np.mean(elbow, axis=0)
	kn = KneeLocator(x, avg, curve='convex', direction='decreasing')
	plt.plot(x, avg)
	plt.xlabel('k')
	plt.ylabel('Distortion')
	plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
	plt.savefig(store4 + 'hblee/4.MPI/4.clustFC/figure/%s-distortion.png' %hemi)
	plt.close()

	elbow = np.load(store4 + 'hblee/4.MPI/4.clustFC/%s-sse.npy' %hemi)
	avg = np.mean(elbow, axis=0)
	kn = KneeLocator(x, avg, curve='convex', direction='decreasing')
	plt.plot(x, avg)
	plt.xlabel('k')
	plt.ylabel('SSE')
	plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
	plt.savefig(store4 + 'hblee/4.MPI/4.clustFC/figure/%s-sse.png' %hemi)
	plt.close()
