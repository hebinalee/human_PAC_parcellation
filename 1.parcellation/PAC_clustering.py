##########################################################
## Perform clustering of PAC using SC
##
## map_t1w     : To create nifti image with given values
## clustering  : To perform clustering based on the SC
## main        : main function to parcel PAC region
##########################################################

import os
from os import listdir
from os.path import isfile, join, exists
import glob
import numpy as np
from mayavi import mlab
import nibabel as nib
from sklearn.cluster import KMeans

basepath = 'X:/path/myfolder/'
datapath = basepath + 'data/'
mainpath = basepath + 'SC_parc/'

## seed ROI index:
## Left : 29(STG), 33(HG), 34(IS)
## Right: 78(STG), 82(HG), 83(IS)

def map_t1w(labels, indices, ref_anat, out_path):
	import copy
	temp_ref = copy.deepcopy(ref_anat)
	temp_vol = np.zeros(temp_ref.shape)
	#temp_sum = matrix.sum(axis=1)[1:]
	#no_conn_vox = np.where(temp_sum > 0, 1,0)
	#temp_vol[seed_idx] += no_conn_vox
	temp_vol[indices] = labels
	img = nib.Nifti1Image(temp_vol, temp_ref.affine)
	nib.save(img, out_path)


def clustering(subID, hemi, n_clusters):
	subpath = join(datapath, subID)
	path_conn = join(subpath, 'tracto')
	path_clust = join(subpath, 'cluster')
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

	clf = KMeans(n_clusters=n_clusters, random_state=0).fit(hemi_conn)
	orig_labels = clf.predict(hemi_conn)
	
	# Manually labels (A --> P)
	#clust_centers = np.zeros((n_clusters, 3))
	#for label in range(n_clusters):
	#	clust_centers[label] = np.mean(hemi_idx[orig_labels == label], axis=0)

	clust = np.zeros(n_clusters)
	for label in range(n_clusters):
		clust[label]  = np.mean(hemi_conn[orig_labels == label, -8])

	sorted_labels = np.zeros_like(orig_labels)
	for new_label, label in enumerate(np.argsort(clust)):
		sorted_labels[orig_labels == label] = new_label
	
	map_t1w(
			sorted_labels + 1,
			tuple(hemi_idx.T),
			ref_anat,
			'%s/%s.clust.K%d.nii.gz' %(path_clust, hemi, n_clusters)
			)
	#return hemi_idx, sorted_labels, hemi_conn, valid_roi


def main(a, b, startname=None):
	sublist = listdir(datapath)
	sublist = sorted(listdir(datapath))
	if startname:
		a = sublist.index(startname)
	
	if b==0:
		sublist = sublist[a:]
	else:
		sublist = sublist[a:b]  # 162026 ~ 793465

	n_clusters = 3
	for sidx, sname in enumerate(sublist):
		print('%dth sub - %s - clustering\n' %(sidx+1, sname))
		clustering(sname, 'L', 4)
		clustering(sname, 'R', 5)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="gradient")
	parser.add_argument(dest="startpoint",type=int,help="Start point of subject for data processing")
	parser.add_argument(dest="endpoint",type=int,help="End point of subject for data processing")
	parser.add_argument("-s",dest="startname",help="The name of the subject to start",required=False)
	args=parser.parse_args()
	main(args.startpoint, args.endpoint, args.startname)
