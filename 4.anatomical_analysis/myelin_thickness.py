#####################################################################################
## To assess difference in myelin density & cortical thickness among PAC subregions
#####################################################################################
'''
[Orders of function implementation]
1) copy_anatomical_data
2) main_compute_mean
3) main_ttest
'''

import os
from os import listdir
from os.path import join, exists, isfile, isdir
import sys
import glob
import shutil
import numpy as np
import nibabel as nib
import scipy.io as sio

import pandas as pd
import matplotlib.pyplot as plt
store6 = '/store6/'
store7 = '/store7/'

Nclusters = 3


'''
[copy_individual_data]
To copy myelin density & cortical thickness file of individual from the Database

Output: 1) /store7/hblee/MPI/data/{subID}/fsaverage_LR32k/{subID}.{hemi}.MyelinMap_BC.32k_fs_LR.func.gii
	2) /store7/hblee/MPI/data/{subID}/fsaverage_LR32k/{subID}.{hemi}.thickness.32k_fs_LR.shape.gii
'''
import zipfile
def copy_individual_data(subID):
	subpath = store7 + 'hblee/MPI/data/' + subID
	outpath = subpath + '/fsaverage_LR32k'

	basepath = store6 + 'Public/Database/HCP_S1200'
	zippath = f'{basepath}/{subID}_3T_Structural_preproc.zip'
	zip_file = zipfile.ZipFile(zippath)
	targets = [f'{subID}/MNINonLinear/fsaverage_LR32k/{subID}.L.MyelinMap_BC.32k_fs_LR.func.gii',
	f'{subID}/MNINonLinear/fsaverage_LR32k/{subID}.L.thickness.32k_fs_LR.shape.gii',
	f'{subID}/MNINonLinear/fsaverage_LR32k/{subID}.R.MyelinMap_BC.32k_fs_LR.func.gii',
	f'{subID}/MNINonLinear/fsaverage_LR32k/{subID}.R.thickness.32k_fs_LR.shape.gii']
	for file in zip_file.namelist():
		for i in range(len(targets)):
			if file.startswith(targets[i]): zip_file.extract(member=file, path=outpath)

	#if exists(join(outpath, target_L)) and exists(join(outpath, target_R)):
	for i in range(len(targets)):
		shutil.move(join(outpath, targets[i]), outpath)
	shutil.rmtree(join(outpath, subID))


'''
[copy_anatomical_data]
To copy myelin density & cortical thickness files from the Database
- copy_individual_data
'''
def copy_anatomical_data():
	datapath = store7 + 'hblee/MPI/data'
	sublist = sorted(listdir(datapath))

	for sidx, subID in enumerate(sublist):
		print(f'{sidx+1}th sub - {subID} - copy data from database\n')
		copy_individual_data(subID)


'''
[compute_mean_values]
To compute averaged myelin density & cortical thickness of PAC subregions for each individual

Input:  1) /store7/hblee/MPI/data/{subID}/6.gradient_cosine/cluster_K3.{hemi}.32k_fs_LR.func.gii
	2) /store7/hblee/MPI/data/{subID}/fsaverage_LR32k/{subID}.{hemi}.MyelinMap_BC.32k_fs_LR.func.gii
	3) /store7/hblee/MPI/data/{subID}/fsaverage_LR32k/{subID}.{hemi}.thickness.32k_fs_LR.shape.gii
Output: 1) mean_myelin - vector of length 3 (0 if Error occured)
	2) mean_thickness - vector of length 3 (0 if Error occured)
'''
def compute_mean_values(subID, hemi):
	# 1) Get cluster label
	subpath = f'{store7}hblee/MPI/data/{subID}'
	labels_file = f'{subpath}/6.gradient_cosine/cluster_K3.{hemi}.32k_fs_LR.func.gii'
	labels = nib.load(labels_file).darrays[0].data    # 32492 vector

	# 2) Get myelin density & thickness
	datapath = subpath + '/fsaverage_LR32k'
	myelin_file = f'{datapath}/{subID}.{hemi}.MyelinMap_BC.32k_fs_LR.func.gii'
	if not isfile(myelin_file):
		print('ERROR: Myelin file not exists!')
		return 0, 0
	myelin = nib.load(myelin_file).darrays[0].data

	thickness_file = f'{datapath}/{subID}.{hemi}.thickness.32k_fs_LR.shape.gii'
	if not isfile(thickness_file):
		print('ERROR: Thickness file not exists!')
		return 0, 0
	thickness = nib.load(thickness_file).darrays[0].data

	# 3) Error if no voxel exists in any of cluster labels
	for i in range(Nclusters):
		if sum(labels == i+1) == 0:
			print(f'ERROR: No voxel in cluster label {i+1}!')
			return 0, 0

	# 4) Compute averaged values
	mean_myelin = []
	mean_thickness = []
	for i in range(Nclusters):
		mean_myelin.append(np.mean(myelin[labels == i+1]))
		mean_thickness.append(np.mean(thickness[labels == i+1]))
	return mean_myelin, mean_thickness
	

'''
[main_compute_mean]
To compute matrix of averaged myelin density & cortical thickness values
- compute_mean_values

Output: /store7/hblee/MPI/2.additional/mean_myelin_thick_{hemi}.mat (N_valid_subj X N_clusters)
'''
def main_compute_mean():
	datapath = store7 + 'hblee/MPI/data'
	sublist = sorted(listdir(datapath))

	for hemi in ['L', 'R']:
		mean_myelin = []
		mean_thickness = []
		for sidx, subID in enumerate(sublist):
			print(f'{sidx+1}th sub - {subID} - calculate mean values {hemi}\n')
			myelin, thickness = compute_mean_values(subID, hemi)
			print(myelin, thickness)
			if not (myelin == 0 and thickness == 0):
				mean_myelin.append(myelin)
				mean_thickness.append(thickness)

		sio.savemat(f'{store7}hblee/MPI/2.additional/mean_myelin_thick_{hemi}.mat', mdict={'mean_myelin': np.array(mean_myelin), 'mean_thickness': np.array(mean_thickness)})
	return


'''
[ttest]
To perform two-sample t-test for each pair of data (FDR correction is applied)

Input:  input_values (N_data_sample X N_cluster)
Output: t, p, corrected_p (print on terminal)
'''
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
def ttest(input_values):
	Nsub = input_values.shape[0]
	
	t = np.zeros(Nclusters)
	p = np.zeros(Nclusters)
	corrected = np.zeros_like(p)
	for i in range(Nclusters):
		t[0], p[0] = ttest_ind(input_values[:,0], input_values[:,1])
		t[1], p[1] = ttest_ind(input_values[:,1], input_values[:,2])
		t[2], p[2] = ttest_ind(input_values[:,2], input_values[:,0])
	
	_, p_corr, _, _ = multipletests(p, 0.05, 'fdr_bh')
	print('t: ', t)
	print('p: ', p)
	print('p_corr: ', p_corr)
	return t, p_corr


'''
[main_ttest]
To perform two-sample t-test in terms of 'myelin density/cortical thickness' for each pair of PAC subregions

Output: t, p, corrected_p (print on terminal)
'''

def main_ttest(hemi):
	mean_values = sio.loadmat(f'{store7}hblee/MPI/2.additional/mean_myelin_thick_{hemi}.mat')
	mean_myelin = mean_values['mean_myelin']
	mean_thickness = mean_values['mean_thickness']
	
	# 1) Myelin density
	print('t-test: Myelin density')
	t, p = ttest(mean_myelin)
	# 2) Cortical thickness
	print('t-test: Cortical thickness')
	t, p = ttest(mean_thickness)
