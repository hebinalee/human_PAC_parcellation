#################################################################################################
## FUNCTIONAL CONNECTIVITY ANALYSIS
##
## regist_clust : To register cluster labels onto the standard volume space
## pearson_conn : To compute the Pearson's correlation coefficient from two time series signals
## meanFC       : To compute cluster X target ROI F matrix in individual-level
## initial_roi  : To return the indices of the initial ROI
#################################################################################################

import os
from os import listdir
from os.path import join, exists, isfile, isdir
import shutil
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

basepath = 'X:/path/myfolder'
datapath = basepath + '/data'

# Cluster results - native(acpc) volume to standard volume
def regist_clust(subID, ncluster):
	subpath = join(datapath, subID)
	standard = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'
	warpfile = join(subpath, 'T1w/acpc_dc2standard.nii.gz')
	path_clust = join(subpath, 'cluster')

	for hemi in ['L', 'R']:
		infile = path_clust + '/%s.clust.K%d.nii.gz' %(hemi, ncluster)
		outfile = path_clust + '/%s.clust.K%d.MNI2mm.nii.gz' %(hemi, ncluster)
		os.system('applywarp --rel --interp=nn -i %s -r %s -w %s -o %s' %(infile, standard, warpfile, outfile))


def pearson_conn(x, y):
	"""Correlate each n with each m.
	x : np.array (N X T)
	y : np.array (M X T)
	Returns : np.array (N X M)
	"""
	mu_x = x.mean(1)
	mu_y = y.mean(1)
	n = x.shape[1]
	if n != y.shape[1]:
		raise ValueError('x and y must have the same number of timepoints.')
	s_x = x.std(1, ddof=n - 1)
	s_y = y.std(1, ddof=n - 1)
	cov = np.dot(x, y.T) - n * np.dot(mu_x[:, np.newaxis], mu_y[np.newaxis, :])
	return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def meanFC(subID):
	ncluster = 3
	subpath = join(datapath, subID)
	path_clust = join(subpath, 'cluster/relabel-SC/')

	fmri_file = join(subpath, 'rsfMRI/rfMRI_REST1_LR.nii.gz')
	if not exists(fmri_file):
		fmri_file = join(subpath, 'rsfMRI/rfMRI_REST1_RL.nii.gz')
	roi_file = join(subpath, 'tracto/roi_MNI2mm.nii.gz')

	fmri = nib.load(fmri_file).get_fdata()
	roi = nib.load(roi_file).get_fdata()

	Nroi = int(roi.max())
	Ntime = fmri.shape[-1]
	roiBOLD = np.zeros((Nroi-2, Ntime))
	
	# ROI-wise averaged BOLD signal
	i = 0
	for ridx in range(Nroi):
		if not (ridx == 32 or ridx == 81):
			roiBOLD[i] = fmri[roi==(ridx+1)].mean(axis=0)
			i += 1

	for hemi in ['L', 'R']:
		meanFC = np.zeros((ncluster, 41))
		path_clust = join(subpath, 'cluster/relabel-SC/')
		clust_file = path_clust + '/%s.clust.K%d.relabel.MNI2mm.nii.gz' %(hemi, ncluster)
		clust = nib.load(clust_file).get_fdata()

		if hemi == 'L':
			target_roi = np.arange(41)
		else:
			target_roi = np.arange(41, 82)

		for label in range(1, ncluster+1):
			idx = np.swapaxes(np.array(np.where(clust == label)), 0, -1)
			if len(idx) != 0:
				voxelBOLD = np.array([fmri[i[0], i[1], i[2], :] for i in idx])
				corr_r = pearson_conn(voxelBOLD, roiBOLD[target_roi, :])
				conn = np.arctanh(((corr_r+1)/2)**6)	# soft thresholding -> r-to-z transform
				#np.save(path_clust + '/FC.K%d.%s.cluster%d.npy' %(ncluster, hemi, label), conn)
				meanFC[label-1, :] = np.mean(conn, axis=0)

		np.save(path_clust + '/meanFC.%s.K%d.npy' %(hemi, ncluster), meanFC)


def initial_roi(subID, hemi):
	ncluster = 3
	subpath = join(datapath, subID)
	roi_file = join(subpath, 'tracto/dil.fs_default.nodes.fixSGM.nii.gz')
	roi = nib.load(roi_file).get_fdata()
	# Create seed ROI
	seed_idx = np.transpose(np.load(join(subpath, 'tracto/fs_default.seed_idx.npy')))
	div_x = np.argmax(np.diff(seed_idx[:,0])) + 1
	n_vox = len(seed_idx)
	hemi_set = np.array(div_x*['R'] + (n_vox-div_x)*['L'])
	for hemi in ['L', 'R']:
		hemi_idx = seed_idx[hemi_set==hemi]
		seed = np.zeros_like(roi)
		for i in range(len(hemi_idx)):
			seed[tuple(hemi_idx[i])] = 1
		if hemi == 'L':
			stg = (roi == 29).astype(int)
			insula = (roi == 34).astype(int)
		else:
			stg = (roi == 78).astype(int)
			insula = (roi == 83).astype(int)
		buf = np.logical_or(stg, insula)
		hg = np.logical_and(seed, ~buf)
		stg = stg*1
		hg = hg*2
		insula = insula*3
		threeroi = stg + hg + insula
	return hg, stg, insula, threeroi


def main(a, b, startname=None):
	sublist = listdir(datapath)
	if startname:
		a = sublist.index(startname)
	
	if b==0:
		sublist = sublist[a:]
	else:
		sublist = sublist[a:b]  # 162026 ~ 793465
	
	ncluster = 3
	for sidx, subID in enumerate(sublist):
		print('%dth sub - %s - FC analysis on cluster\n' %(sidx+1, subID))
		meanFC(subID)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="gradient")
	parser.add_argument(dest="startpoint",type=int,help="Start point of subject for data processing")
	parser.add_argument(dest="endpoint",type=int,help="End point of subject for data processing")
	parser.add_argument("-s",dest="startname",help="The name of the subject to start",required=False)
	args=parser.parse_args()
	main(args.startpoint, args.endpoint, args.verison, args.startname)
