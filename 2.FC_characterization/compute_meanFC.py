######################################
## FUNCTIONAL CONNECTIVITY ANALYSIS ##
######################################

import os
from os import listdir
from os.path import join, exists, isfile, isdir
import shutil
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

if os.name == 'nt':
	store4 = 'X:/'
else:
	store4 = '/store4/'


# Cluster results - native(acpc) volume to standard volume
def regist_clust(subID, ncluster, version):
	subpath = join(store4, 'hblee/4.MPI/4.clustFC/data', subID)
	standard = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'
	warpfile = join(subpath, 'T1w/acpc_dc2standard.nii.gz')
	path_clust = join(subpath, 'cluster%d' %version)

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


def meanFC(subID, version):
	ncluster = 3
	subpath = join(store4, 'hblee/4.MPI/4.clustFC/data', subID)
	path_clust = join(subpath, 'cluster%d/relabel-SC/' %version)

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
		if version == 5:
			if hemi == 'L':
				ncluster = 4
			else:
				ncluster = 5
		meanFC = np.zeros((ncluster, 41))
		if version == 3:
			path_clust = join(subpath, 'cluster%d/relabel-SC4/' %version)
		else:
			path_clust = join(subpath, 'cluster%d/relabel-SC/' %version)
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


def plot_fc(subID, hemi, V_cluster):
	ncluster = 3
	subpath = join(store4, 'hblee/4.MPI/4.clustFC/data', subID)
	path_clust = join(subpath, 'cluster%d/relabel-SC4' %V_cluster)
	outpath = store4 + 'hblee/4.MPI/4.clustFC/figure/fc%d/' %V_cluster

	#if not exists(outpath + subID + '-' + hemi + '.png'):
	conn1 = np.load(path_clust + '/FC.K%d.%s.cluster1.npy' %(ncluster, hemi))
	conn2 = np.load(path_clust + '/FC.K%d.%s.cluster2.npy' %(ncluster, hemi))
	conn3 = np.load(path_clust + '/FC.K%d.%s.cluster3.npy' %(ncluster, hemi))
	fig = plt.figure()
	ax1 = fig.add_subplot(3, 1, 1)
	ax2 = fig.add_subplot(3, 1, 2)
	ax3 = fig.add_subplot(3, 1, 3)
	#axes = (ax1, ax2, ax3)
	plt.subplots_adjust(left=0.1, right=0.9)
	ax1.imshow(conn1)
	ax2.imshow(conn2)
	ax3.imshow(conn3)
	resize(axes)
	#fig.set_figwidth(20)
	#fig.colorbar(a, ax=ax1)
	#fig.colorbar(b, ax=ax2)
	#fig.colorbar(c, ax=ax3)
	#plt.show()
	fig.savefig(outpath + subID + '-' + hemi + '.png')


def plot_fc2(subID, hemi, V_cluster):
	ncluster = 3
	subpath = join(store4, 'hblee/4.MPI/4.clustFC/data', subID)
	path_clust = join(subpath, 'cluster%d/relabel-SC4' %V_cluster)
	outpath = store4 + 'hblee/4.MPI/4.clustFC/figure/fc%d/' %V_cluster

	#if not exists(outpath + subID + '-' + hemi + '.png'):
	conn = [None]*ncluster
	fig, axes = plt.subplots(nrows=3, ncols=1)
	#plt.subplots_adjust(left=0.1, right=0.9)
	for i in range(ncluster):
		conn[i] = np.load(path_clust + '/FC.K%d.%s.cluster%d.npy' %(ncluster, hemi, i+1))
		axes[i].imshow(conn[i])
		f = axes[i].figure
	#resize(axes)
	#fig.tight_layout()
	#fig.set_figwidth(20)
	#fig.subplots_adjust(wspace=0.8, hspace=0.6)
	fig.savefig(outpath + hemi + '-' + subID + '.png')


def resize(axes):
	# this assumes a fixed aspect being set for the axes.
	for ax in axes:
		width = np.diff(ax.get_xlim())[0]
		height = -np.diff(ax.get_ylim())[0]
		fig = ax.figure
		fixed = 30
		fig.set_size_inches(fixed*height/width, fixed)
		#fig.set_size_inches(fixed, fixed*height/width)


def plot_meanfc(subID, hemi, V_cluster, V_label):
	ncluster = 3
	subpath = join(store4, 'hblee/4.MPI/4.clustFC/data', subID)
	path_clust = join(subpath, 'cluster%d' %V_cluster, 'relabel-SC%d/' %V_label)
	if V_label == 0:
		path_clust = join(subpath, 'cluster%d/relabel-SC/' %V_cluster)
	outpath = store4 + 'hblee/4.MPI/4.clustFC/figure/fc%d/' %V_cluster

	#if not exists(outpath + subID + '-' + hemi + '.png'):
	if exists(path_clust + '/meanFC.%s.K%d.npy' %(hemi, ncluster)):
		conn = np.load(path_clust + '/meanFC.%s.K%d.npy' %(hemi, ncluster))
		plt.imshow(conn)
		plt.savefig(outpath + hemi + '-' + subID + '.png')


def initial_roi(subID, hemi):
	ncluster = 3
	subpath = join(store4, 'hblee/4.MPI/4.clustFC/data', subID)
	# roi_file = join(subpath, 'tracto/roi_MNI2mm.nii.gz')
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

		# Using mask of valid_idx
		clust_file = subpath + '/cluster1/%s.clust.K3.KMeans.nii.gz' %hemi
		clust = nib.load(clust_file).get_fdata()
		validroi = threeroi * (clust > 0)
		np.unique(validroi)
	return hg, stg, insula, threeroi, validroi


def main(a, b, version, startname=None):
	datapath = store4 + 'hblee/4.MPI/4.clustFC/data/'
	sublist = listdir(datapath)
	if startname:
		a = sublist.index(startname)
	
	if b==0:
		sublist = sublist[a:]
	else:
		sublist = sublist[a:b]  # 162026 ~ 793465
	
	ncluster = 3
	basepath = join(store4, 'hblee/4.MPI/4.clustFC/figure/fc3/')
	for sidx, sname in enumerate(sublist):
		print('%dth sub - %s - FC analysis on cluster\n' %(sidx+1, sname))
		#regist_clust(sname, ncluster, version)
		#fc(sname, 4)
		meanFC(sname, version)
		#if exists(join(store4, 'hblee/4.MPI/4.clustFC/data', sname, 'cluster3/relabel-SC4/meanFC.L.K3.npy')):
		#	plot_fc2(sname, 'L', version)
		#if exists(join(store4, 'hblee/4.MPI/4.clustFC/data', sname, 'cluster3/relabel-SC4/meanFC.R.K3.npy')):
		#	plot_fc2(sname, 'R', version)
		#plot_meanfc(sname, 'L', version, 0)
		#plot_meanfc(sname, 'R', version, 0)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="gradient")
	parser.add_argument(dest="startpoint",type=int,help="Start point of subject for data processing")
	parser.add_argument(dest="endpoint",type=int,help="End point of subject for data processing")
	parser.add_argument(dest="version",type=int,help="Trial")
	parser.add_argument("-s",dest="startname",help="The name of the subject to start",required=False)
	args=parser.parse_args()
	main(args.startpoint, args.endpoint, args.verison, args.startname)
