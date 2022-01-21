###############################################################################################
## To perform gradient analysis
##
## calculate_sim  : To compute simmularity matrix from FC matrix
## gradient       : To perform gradient analysis in individual-level
## group_average  : To compute the group-averaged gradient
## align_gradient : To align individual-level gradient onto the group-averaged gradient space
###############################################################################################
'''
[Order of function implementation]
1) gradient
2) group_average
3) align_gradient
'''
import os
from os import listdir
from os.path import join, exists, isfile, isdir
import sys
import glob
import shutil
import scipy.io as sio
import numpy as np
import nibabel as nib
import zipfile

basepath = 'X:/path/myfolder'
datapath = basepath + '/data'

sys.path.append(basepath + '/congrads-master')
import conmap_surf2, conmap_sim

def set_subpath(subID): return f'{datapath}/{subID}'
def set_inpath(subID): return f'{datapath}/{subID}/seed_surf'
def set_outpath(subID): return f'{datapath}/{subID}/gradient'



'''
[calculate_sim]
To compute cosine simularity matrix from correlation matrix

Input:  1) {subpath}/gradient/merged_seed.{hemi}.32k_fs_LR.correlation1.mat (N_seed_voxels X N_target_voxels)
	2) {subpath}/gradient/merged_seed.{hemi}.32k_fs_LR.correlation2.mat (N_seed_voxels X N_target_voxels)
Output: S (N_seed_voxels X N_seed_voxels)
'''
from brainspace.gradient.kernels import compute_affinity
def calculate_sim(subID, hemi):
	inpath = set_inpath(subID)
	x = sio.loadmat(f'{inpath}/merged_seed.{hemi}.32k_fs_LR.correlation1.mat')['R']
	S = compute_affinity(x, kernel='cosine', sparsity=0)
	x = sio.loadmat(f'{inpath}/merged_seed.{hemi}.32k_fs_LR.correlation2.mat')['R']
	S += compute_affinity(x, kernel='cosine', sparsity=0)
	S /= 2
	return S


'''
[gradient]
To perform gradient analysis for all subjects
- calculate_sim

Input:  S - List of simularity maps for all subjects
Output: {subpath}/gradient/merged_seed.{hemi}.32k_fs_LR.gradient.mat
* All I/Os are on the fsaverage_LR32k surface space
'''
def gradient(hemi):
	sublist = sorted(listdir(datapath))
	
	S = []
	for sidx, subID in enumerate(sublist):
		S.append(calculate_sim(subID, hemi))

	print('Gradien analysis: STARTED')
	from brainspace.gradient.gradient import GradientMaps
	GM = GradientMaps(n_components=10, approach='pca', kernel=None, alignment=None, random_state=None)
	GM.fit(S, gamma=None, sparsity=0, n_iter=10, reference=None)
	gradients = GM.gradients_
	lambdas = GM.lambdas_

	print('Gradien analysis: DONE')
	for sidx, subID in enumerate(sublist):
		outpath = set_outpath(subID)
		if not exists(outpath): os.makedirs(outpath)
		sio.savemat(f'{outpath}/merged_seed.{hemi}.32k_fs_LR.gradient.mat', mdict={'gradient':gradients[sidx], 'lambda':lambdas[sidx]})


'''
[group_average]
To compute group averaged gradient data by performing PCA on stacks of individual data

Input:  {subpath}/gradient/merged_seed.{hemi}.32k_fs_LR.gradient.mat
Output: {basepath}/gradient/merged_seed.{hemi}.32k_fs_LR.mean_gradient.mat
* All I/Os are on the fsaverage_LR32k surface space
'''
def group_average(hemi):
	sublist = sorted(listdir(datapath))
	
	for sidx, subID in enumerate(sublist):
		outpath = set_outpath(subID)
		x = sio.loadmat(f'{outpath}/merged_seed.{hemi}.32k_fs_LR.gradient.mat')
		if not sidx:
			X = x['gradient']
		else:
			X = np.hstack((X, x['gradient']))
	print('Shape of X: ', X.shape)

	from brainspace.gradient.embedding import PCAMaps
	PM = PCAMaps(n_components=10, random_state=None)
	PM.fit(X)
	X_ref = PM.maps_
	print('Shape of X after PCA: ', X_ref.shape)
	sio.savemat(f'{basepath}/gradient/merged_seed.{hemi}.32k_fs_LR.mean_gradient.mat', mdict={'grad_ref':X_ref})


'''
[align_gradient]
To align individual gradient results using procrustes alignment algorithm

Input:  1) {subpath}/gradient/merged_seed.{hemi}.32k_fs_LR.gradient.mat
	2) {basepath}/gradient/merged_seed.{hemi}.32k_fs_LR.mean_gradient.mat
Output: {subpath}/gradient/merged_seed.{hemi}.32k_fs_LR.gradient.aligned.mat
* All I/Os are on the fsaverage_LR32k surface space
'''
from brainspace.gradient.alignment import ProcrustesAlignment
def align_gradient(hemi):
	PA = ProcrustesAlignment(n_iter=10)

	sublist = sorted(listdir(datapath))
	
	X = []
	for sidx, subID in enumerate(sublist):
		outpath = set_outpath(subID)
		x = sio.loadmat(f'{outpath}/merged_seed.{hemi}.32k_fs_LR.gradient.mat')['gradient']
		X.append(x)

	ref = sio.loadmat(f'{basepath}/gradient/merged_seed.{hemi}.32k_fs_LR.mean_gradient.mat')['grad_ref']
	PA.fit(X, reference=ref)
	aligned = PA.aligned_

	for sidx, subID in enumerate(sublist):
		outpath = set_outpath(subID)
		sio.savemat(f'{outpath}/merged_seed.{hemi}.32k_fs_LR.gradient.aligned.mat', mdict={'gradient':aligned[sidx]})


'''
[main]
Main function to perform analysis
'''
def main(a, b, hemi='L', startname=None):
	sublist = sorted(listdir(datapath))
	if startname:
		a = sublist.index(startname)
	if b==0:
		sublist = sublist[a:]
	else:
		sublist = sublist[a:b]  # 162026 ~ 793465
	
	# 1. gradient(hemi)
	# 2. group_average(hemi)
	# 3. align_gradient(hemi)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="gradient")
	parser.add_argument(dest="startpoint",type=int,help="Start point of subject for data processing")
	parser.add_argument(dest="endpoint",type=int,help="End point of subject for data processing")
	parser.add_argument(dest="hemi",type=str, help="Hemisphere to perform analysis on")
	parser.add_argument("-s",dest="startname",help="The name of the subject to start",required=False)
	args=parser.parse_args()
	main(args.startpoint, args.endpoint, args.startname)
