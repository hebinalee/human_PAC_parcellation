##################################################################################################
## TO CONDUCT STATISTICAL TEST
##
## dilate_seed          : To dilate valid_seed and map onto MNI_2mm space(vol) and surface space
## manual_interpolation : To perform iterative dilation of valid_seed region
## save_norm_img        : To normalize gradient and save into GIFTI file
## calculate_mean_grad  : To compute averaged gradient of clusters for each individual
## save_mean_grad       : To save averaged gradient values
## ttest                : To perform two-sample t-test
##################################################################################################
sssssssssssssssssssssssssss
'''
[Order of function implementation]
1) dilate_seed
2) save_norm_img
3) save_mean_grad
4) ttest
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
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

basepath = 'X:/path/myfolder'
datapath = basepath + '/data'

sys.path.append(basepath + '/congrads-master')
import conmap_surf2, conmap_sim

def set_subpath(subID): return f'{datapath}/{subID}'
def set_inpath(subID): return f'{datapath}/{subID}'
def set_outpath(subID): return f'{datapath}/{subID}/gradient'

Nclusters = 3



'''
[dilate_seed]
To assign label to all seed voxels
and register labeled seed area onto fsaverage_LR32k surface space (via standard volume space)
- manual_interpolation

Input:  1) {subpath}/cluster/relabel/{hemi}.clust.K3.relabel.nii.gz
	2) {subpath}/gradient/seed_{hemi}.nii.gz
Output: 1) {subpath}/gradient/{hemi}.clust.K3.dilated.nii.gz
	2) {subpath}/gradient/{hemi}.clust.K3.dilated.MNI2mm.nii.gz
	3) {subpath}/gradient/cluster_K3.{hemi}.32k_fs_LR.func.gii
'''
def dilate_seed(subID, hemi):
	subpath = set_subpath(subID)
	inpath = set_inpath(subID)
	outpath = set_outpath(subID)
	
	labels_dir = f'{inpath}/cluster/relabel-SC/{hemi}.clust.K3.relabel.nii.gz'
	seed_dir = f'{outpath}/seed_{hemi}.nii.gz'
	output_dir = f'{outpath}/{hemi}.clust.K3.dilated.nii.gz'
	ref_img = nib.load(labels_dir)
	labels = ref_img.get_fdata()
	seed = nib.load(seed_dir).get_fdata()

	# 1) Dilate labels and save image file
	out_labels = manual_interpolation(labels, seed)
	img = nib.Nifti1Image(out_labels, ref_img.affine)
	nib.save(img, output_dir)
	
	# 2) Register to volume MNI2mm space
	standard = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'
	warpfile = f'{inpath}/T1w/acpc_dc2standard.nii.gz'
	output_MNI_dir = f'{outpath}/{hemi}.clust.K3.dilated.MNI2mm.nii.gz'
	os.system(f'applywarp --rel --interp=nn -i {output_dir} -r {standard} -w {warpfile} -o {output_MNI_dir}')
	
	# 3) Map onto surface space
	surf_dir = f'{subpath}/fsaverage_LR32k/{subID}.{hemi}.midthickness.32k_fs_LR.surf.gii'
	output_surf_dir = f'{outpath}/cluster_K3.{hemi}.32k_fs_LR.func.gii'
	os.system(f'wb_command -volume-to-surface-mapping {output_MNI_dir} {surf_dir} {output_surf_dir} -enclosing')


'''
[manual_interpolation]
To assign label to all seed voxels
(originally, labels are assigned only for valid voxels)

Input:  1) input_data - Label data for valid seed voxels
	2) target - Seed ROI data
Output: input_data - Label data for whole seed 
* All I/Os are on the native volume space
'''
from collections import Counter
def manual_interpolation(input_data, target):
	changed = 0
	while sum(sum(sum(input_data!=0))) < sum(sum(sum(target!=0))) and changed != 0:
		changed = 0
		search_range = np.where(np.logical_and(target, input_data==0))
		for i in range(search_range):
			index = tuple(np.transpose(np.array(search_range))[i])
			c = Counter(input_data[index[0]-1:index[0]+2, index[1]-1:index[1]+2, index[2]-1:index[2]+2].flatten())
			c.pop(0)
			# if all of neighbors have value of 0
			if not c:
				input_data[index] = 0 
			# priority: 2, 1, 3
			elif len(c) == 3 and c[1]==c[2]==c[3]:
				input_data[index] = 2
				changed += 1
			elif len(c) >= 2 and c.most_common(2)[0][1] == c.most_common(2)[1][1]:
				a = c.most_common(2)[0][0]
				b = c.most_common(2)[1][0]
				changed += 1
				if a == 2 or b == 2:
					input_data[index] = 2
				else:
					input_data[index] = 1
			else:
				input_data[index] = c.most_common(1)[0][0]
				changed += 1
	return input_data


'''
[save_norm_img]
To normalize gradient data and crop individual seed area (normalize -> crop)

Input:  1) {basepath}/gradient/merged_seed.{hemi}.32k_fs_LR.func.gii
	2) {subpath}/gradient/seed_{hemi}.32k_fs_LR.func.gii
	3) {subpath}/gradientv/merged_seed.{hemi}.32k_fs_LR.gradient.aligned.mat
Output: {subpath}/gradient/merged_seed.{hemi}.32k_fs_LR.cmaps.aligned.norm.crop.func.gii
* All I/Os are on the fsaverage_LR32k surface space
'''
def save_norm_img(subID, hemi, nmaps=3):
	subpath = set_subpath(subID)
	outpath = set_outpath(subID)

	merged_seed_img = nib.load(f'{basepath}/gradient/merged_seed.{hemi}.32k_fs_LR.func.gii')
	merged_seed = merged_seed_img.darrays[0].data
	mergedSeedIndices = np.where(merged_seed==1)

	seed_img = nib.load(f'{outpath}/seed_{hemi}.32k_fs_LR.func.gii')
	seed = seed_img.darrays[0].data
	maskIndices = np.where(seed==0)

	ref_img = nib.load(f'{outpath}/merged_seed.{hemi}.32k_fs_LR.cmaps.aligned.crop.func.gii')
	gradient = sio.loadmat(f'{outpath}/merged_seed.{hemi}.32k_fs_LR.gradient.aligned.mat')['gradient']

	for i in range(nmaps):
		dat = np.zeros_like(merged_seed)
		dat[mergedSeedIndices] = (gradient[:,i] - np.mean(gradient[:,i])) / np.std(gradient[:,i])
		dat[maskIndices] = 0
		ref_img.darrays[i].data = dat
	
	outfile = f'{outpath}/merged_seed.{hemi}.32k_fs_LR.cmaps.aligned.norm.crop.func.gii'
	nib.save(ref_img, outfile)


'''
[calculate_mean_grad]
To calculate mean gradient values for 3 PAC subregions

Input:  1) {subpath}/gradient/cluster_K3.{hemi}.32k_fs_LR.func.gii
	2) {subpath}/gradient/merged_seed.{hemi}.32k_fs_LR.cmaps.aligned.norm.crop.func.gii
Output: mean_gradient - Vector with length 3 (If failed, return None)
* All I/Os are on the fsaverage_LR32k surface space
'''
def calculate_mean_grad(subID, hemi):
	subpath = set_subpath(subID)
	outpath = set_outpath(subID)
	if exists(f'{outpath}/cluster_K3.{hemi}.32k_fs_LR.label.gii'):
		os.remove(f'{outpath}/cluster_K3.{hemi}.32k_fs_LR.label.gii')

	labels = nib.load(f'{outpath}/cluster_K3.{hemi}.32k_fs_LR.func.gii').darrays[0].data
	gradient = nib.load(f'{outpath}/merged_seed.{hemi}.32k_fs_LR.cmaps.aligned.norm.crop.func.gii').darrays[0].data

	mean_gradient = []
	for i in range(Nclusters):
		label_grad = gradient[labels == i+1]
		nonzero_grad = label_grad[label_grad != 0]
		# If all values are zero
		if nonzero_grad.size == 0:
			return None
		mean_grad = np.mean(nonzero_grad)
		mean_gradient.append(mean_grad)

	return mean_gradient


'''
[save_mean_grad]
To compute mean gradient values for 3 PAC subregions for all individials and return the matrix
- calculate_mean_grad

Input:
Output: {basepath}/gradient/mean_gradient_{hemi}.mat (N_valid_subj X N_clusters)
'''
def save_mean_grad():
	sublist = sorted(listdir(datapath))

	for hemi in ['L', 'R']:
		mean_gradient = []
		for sidx, subID in enumerate(sublist):
			print('%dth sub - %s - calculate mean gradient %s\n' %(sidx+1, subID, hemi))
			results = calculate_mean_grad(subID, hemi)
			if results:
				mean_gradient.append(results)

		sio.savemat(f'{basepath}/gradient/mean_gradient_{hemi}.mat', mdict={'mean_grad': np.array(mean_gradient)})
	return


'''
[ttest]
To perform two-sample t-test for each pair of PAC subregions (FDR correction is applied)

Input:  {basepath}/gradient/mean_gradient_{hemi}.mat
Output: t, p, corrected_p (print on terminal)
'''
def ttest(hemi):
	mean_grad = sio.loadmat(f'{basepath}/gradient/mean_gradient_{hemi}.mat')['mean_grad']
	Nsub = mean_grad.shape[0]
	
	t = np.zeros(Nclusters)
	p = np.zeros(Nclusters)
	corrected = np.zeros_like(p)
	for i in range(Nclusters):
		t[0], p[0] = ttest_ind(mean_grad[:,0], mean_grad[:,1])
		t[1], p[1] = ttest_ind(mean_grad[:,1], mean_grad[:,2])
		t[2], p[2] = ttest_ind(mean_grad[:,2], mean_grad[:,0])
	
	_, p_corr, _, _ = multipletests(p, 0.05, 'fdr_bh')
	print('t: ', t)
	print('p: ', p)
	print('p_corr: ', p_corr)


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
	
	# 3. save_mean_grad()
	# 4. ttest(hemi)
	for sidx, subID in enumerate(sublist):
		print('f'{sidx+1}th sub - {subID} - post-hoc anaylsis {hemi}\n')
		# 1. dilate_seed(subID, hemi)
		# 2. save_norm_img(subID, hemi)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="gradient")
	parser.add_argument(dest="startpoint",type=int,help="Start point of subject for data processing")
	parser.add_argument(dest="endpoint",type=int,help="End point of subject for data processing")
	parser.add_argument(dest="hemi",type=str, help="Hemisphere to perform analysis on")
	parser.add_argument("-s",dest="startname",help="The name of the subject to start",required=False)
	args=parser.parse_args()
	main(args.startpoint, args.endpoint, args.startname)
