#############################################################################################
## To access difference in myelin density & cortical thickness among PAC subregions
##
## compute_mean_values : To compute average myelin density & cortical thickness values
## main_compute_mean   : To obtain vectors of average values of whole participants
## ttest               : To perform two-sample t-test
## main_ttest          : To compute group difference in myelin density & cortical thickness
#############################################################################################


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

Nclusters = 3


'''''
Step 1) Compute average myelin density & cortical thickness values
'''''
def compute_mean_values(subID, hemi):
	# Get cluster labels
	subpath = f'{store7}hblee/MPI/data/{subID}'
	labels_file = f'{subpath}/6.gradient_cosine/cluster_K3.{hemi}.32k_fs_LR.func.gii'
	labels = nib.load(labels_file).darrays[0].data    # 32492 vector
  
  # Do not compute average values if no voxel assigned in cluster label
	for i in range(Nclusters):
		if sum(labels == i+1) == 0:
			print(f'ERROR: No voxel in cluster label {i+1}!')
			return 0, 0

	# Get myelin density & thickness
	datapath = subpath + '/fsaverage_LR32k'
	myelin_file = f'{datapath}/{subID}.{hemi}.MyelinMap_BC.32k_fs_LR.func.gii'
	myelin = nib.load(myelin_file).darrays[0].data
	thickness_file = f'{datapath}/{subID}.{hemi}.thickness.32k_fs_LR.shape.gii'
	thickness = nib.load(thickness_file).darrays[0].data

  # Compute averaged values of subregions
	mean_myelin = []
	mean_thickness = []
	for i in range(Nclusters):
		mean_myelin.append(np.mean(myelin[labels == i+1]))
		mean_thickness.append(np.mean(thickness[labels == i+1]))
	return mean_myelin, mean_thickness


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


'''''
Step 2) Group difference statistical test
'''''
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


def main_ttest(hemi):
	mean_values = sio.loadmat(f'{store7}hblee/MPI/2.additional/mean_myelin_thick_{hemi}.mat')
	mean_myelin = mean_values['mean_myelin']
	mean_thickness = mean_values['mean_thickness']
	
	# Myelin density
	print('t-test: Myelin density')
	t, p = ttest(mean_myelin)
	# Cortical thickness
	print('t-test: Cortical thickness')
	t, p = ttest(mean_thickness)
