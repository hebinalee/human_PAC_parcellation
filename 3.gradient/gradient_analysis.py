###############################################################################################
## To perform gradient analysis
##
## calculate_sim  : To compute simmularity matrix from FC matrix
## gradient       : To perform gradient analysis in individual-level
## group_average  : To compute the group-averaged gradient
## align_gradient : To align individual-level gradient onto the group-averaged gradient space
## save_img       : To save gifti image file including gradient informations
###############################################################################################

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

from brainspace.gradient.kernels import compute_affinity
def calculate_sim(subID, hemi):
	inpath = set_inpath(subID)
	x = sio.loadmat(f'{inpath}/merged_seed.{hemi}.32k_fs_LR.correlation1.mat')['R']
	S = compute_affinity(x, kernel='cosine', sparsity=0)
	x = sio.loadmat(f'{inpath}/merged_seed.{hemi}.32k_fs_LR.correlation2.mat')['R']
	S += compute_affinity(x, kernel='cosine', sparsity=0)
	S /= 2
	return S


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
	sio.savemat(f'{store7}hblee/MPI/1.gradient/merged_seed.{hemi}.32k_fs_LR.mean_gradient6.mat', mdict={'grad_ref':X_ref})


def align_gradient(hemi):
	from brainspace.gradient.alignment import ProcrustesAlignment
	PA = ProcrustesAlignment(n_iter=10)

	sublist = sorted(listdir(datapath))
	
	X = []
	for sidx, subID in enumerate(sublist):
		outpath = set_outpath(subID)
		x = sio.loadmat(f'{outpath}/merged_seed.{hemi}.32k_fs_LR.gradient.mat')['gradient']
		X.append(x)

	ref = sio.loadmat(f'{store7}hblee/MPI/1.gradient/merged_seed.{hemi}.32k_fs_LR.mean_gradient6.mat')['grad_ref']
	PA.fit(X, reference=ref)
	aligned = PA.aligned_

	for sidx, subID in enumerate(sublist):
		outpath = set_outpath(subID)
		sio.savemat(f'{outpath}/merged_seed.{hemi}.32k_fs_LR.gradient.aligned.mat', mdict={'gradient':aligned[sidx]})


def save_img(subID, hemi, nmaps=3):
	subpath = set_subpath(subID)
	inpath = set_inpath(subID)
	outpath = set_outpath(subID)

	merged_seed_img = nib.load(f'{store7}hblee/MPI/1.gradient/merged_seed.{hemi}.32k_fs_LR.func.gii')
	merged_seed = merged_seed_img.darrays[0].data
	mergedSeedIndices = np.where(merged_seed==1)

	seed_img = nib.load(f'{inpath}/seed_{hemi}.32k_fs_LR.func.gii')
	seed = seed_img.darrays[0].data
	maskIndices = np.where(seed==0)

	ref_img = nib.load(f'{inpath}/merged_seed.{hemi}.32k_fs_LR.cmaps.aligned.crop.func.gii')
	gradient = sio.loadmat(f'{outpath}/merged_seed.{hemi}.32k_fs_LR.gradient.aligned.mat')['gradient']

	for i in range(nmaps):
		dat = np.zeros_like(merged_seed)
		dat[mergedSeedIndices] = gradient[:,i]
		dat[maskIndices] = 0
		ref_img.darrays[i].data = dat
	
	outfile = f'{outpath}/merged_seed.{hemi}.32k_fs_LR.cmaps.aligned.crop.func.gii'
	nib.save(ref_img, outfile)


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
	for sidx, subID in enumerate(sublist):
		print('%dth sub - %s - gradient anaylsis %s\n' %(sidx+1, subID, hemi))
		# 4. save_img(subID, hemi)
		#for hemi in ['L', 'R']:
		#	crop_seed(subID, hemi)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="gradient")
	parser.add_argument(dest="startpoint",type=int,help="Start point of subject for data processing")
	parser.add_argument(dest="endpoint",type=int,help="End point of subject for data processing")
	parser.add_argument(dest="hemi",type=str, help="Hemisphere to perform analysis on")
	parser.add_argument("-s",dest="startname",help="The name of the subject to start",required=False)
	args=parser.parse_args()
	main(args.startpoint, args.endpoint, args.startname)
