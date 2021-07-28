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

if os.name == 'nt':
	#store4 = 'X:/'
	store6 = 'Y:/'
	store7 = 'V:/'
else:
	#store4 = '/store4/'
	store6 = '/store6/'
	store7 = '/store7/'
sys.path.append(store7 + 'hblee/MPI/1.gradient/congrads-master')
import conmap_surf2, conmap_sim

def set_subpath(subID): return store7 + 'hblee/MPI/data/' + subID
def set_inpath(subID): return store7 + 'hblee/MPI/data/' + subID + '/4.gradient_new'
def set_outpath(subID): return store7 + 'hblee/MPI/data/' + subID + '/6.gradient_cosine'

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
	datapath = store7 + 'hblee/MPI/data'
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
	datapath = store7 + 'hblee/MPI/data'
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

	datapath = store7 + 'hblee/MPI/data'
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


def copy_results(hemi):
	datapath = store7 + 'hblee/MPI/data'
	sublist = sorted(listdir(datapath))
	outpath = store7 + 'hblee/MPI/1.gradient/results6'
	for sidx, subID in enumerate(sublist):
		subpath = set_outpath(subID)
		resultfile = subpath + '/merged_seed.%s.32k_fs_LR.cmaps.aligned.crop.func.gii' %hemi
		shutil.copy(resultfile, f'{outpath}/{subID}.{hemi}.func.gii')


def main(a, b, hemi='L', startname=None):
	datapath = store7 + 'hblee/MPI/data'
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
