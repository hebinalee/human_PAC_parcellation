#####################################
## FUNCTIONAL CONNECTIVITY ANALYSIS
#####################################
'''
[Order of function implementation]
1) compute_meanFC
'''
import os
from os import listdir
from os.path import join, exists, isfile, isdir
import shutil
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

if os.name == 'nt':
	store7 = 'V:/'
else:
	store7 = '/storeV/'

Nclusters = 3



'''
[pearson_conn]
To compute connectivity matrix from two data arrays using pearson's correlation

Input:  1) x (N X T)
		2) y (M X T)
Output: correlation matrix (N X M)
'''
def pearson_conn(x, y):
	mu_x = x.mean(1)
	mu_y = y.mean(1)
	n = x.shape[1]
	if n != y.shape[1]:
		raise ValueError('x and y must have the same number of timepoints.')
	s_x = x.std(1, ddof=n - 1)
	s_y = y.std(1, ddof=n - 1)
	cov = np.dot(x, y.T) - n * np.dot(mu_x[:, np.newaxis], mu_y[np.newaxis, :])
	return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


'''
[compute_meanFC]
To compute averaged FC values for each clusters
- pearson_conn

Input:  1) /store7/hblee/MPI/data1/{subID}/rsfMRI/rfMRI_REST1_LR.nii.gz
		2) /store7/hblee/MPI/data1/{subID}/tracto/roi_MNI2mm.nii.gz
		3) /store7/hblee/MPI/data1/{subID}/cluster4/relabel-SC/{hemi}.clust.K3.relabel.MNI2mm.nii.gz
* All inputs are on the standard volume space (MNI2mm)
Output: /store7/hblee/MPI/data/{subID}/cluster/meanFC.{hemi}.K3.npy (3 X 82)
* Do not use this /store7/hblee/MPI/data1/{subID}/cluster4/relabel-SC/meanFC.{hemi}.K3.npy (3 X 41)
'''
def compute_meanFC(subID):
	subpath = join(store4, 'hblee/MPI/data1', subID)
	path_clust = join(subpath, 'cluster4/relabel-SC/')
	outpath = f'{store7}hblee/MPI/data/{subID}/cluster'

	# 1) Load fMRI and ROI data
	fmri_file = join(subpath, 'rsfMRI/rfMRI_REST1_LR.nii.gz')
	if not exists(fmri_file):
		fmri_file = join(subpath, 'rsfMRI/rfMRI_REST1_RL.nii.gz')
	roi_file = join(subpath, 'tracto/roi_MNI2mm.nii.gz')

	fmri = nib.load(fmri_file).get_fdata()
	roi = nib.load(roi_file).get_fdata()

	Nroi = int(roi.max())
	Ntime = fmri.shape[-1]
	roiBOLD = np.zeros((Nroi-2, Ntime))
	
	# 2) Compute ROI-wise averaged BOLD signal
	i = 0
	for ridx in range(Nroi):
		if not (ridx == 32 or ridx == 81):
			roiBOLD[i] = fmri[roi==(ridx+1)].mean(axis=0)
			i += 1

	for hemi in ['L', 'R']:
		meanFC = np.zeros((Nclusters, 41))
		path_clust = join(subpath, 'cluster4/relabel-SC/')
		clust_file = f'{path_clust}/{hemi}.clust.K{Nclusters}.relabel.MNI2mm.nii.gz'
		clust = nib.load(clust_file).get_fdata()

		for label in range(Nclusters):
			idx = np.swapaxes(np.array(np.where(clust == label+1)), 0, -1)
			if len(idx) != 0:
				voxelBOLD = np.array([fmri[i[0], i[1], i[2], :] for i in idx])
				corr_r = pearson_conn(voxelBOLD, roiBOLD)
				conn = np.arctanh(((corr_r+1)/2)**6)	# soft thresholding -> r-to-z transform
				meanFC[label, :] = np.mean(conn, axis=0)

		np.save(f'{outpath}/meanFC.{hemi}.K{Nclusters}.npy', meanFC)

'''
[roiconn]
To save matrix of meanFC values for each ROI (N_valid_subj X N_clusters)

Input:  /store7/hblee/MPI/data1/{subID}/cluster4/relabel-SC/meanFC.{hemi}.K3.npy (3 X 82)
Output: /store7/hblee/MPI/stat/roiFC/{hemi}-ROI{i}.npy
'''
def roiconn(hemi):
	datapath = store7 + 'hblee/MPI/data/'
	sublist = sorted(listdir(datapath))
	Nsub = len(sublist)
	Nroi = 82

	meanconn = np.zeros((Nsub, Nclusters, Nroi))
	i = 0
	for sidx, subID in enumerate(sublist):
		subpath = f'{store7}hblee/MPI/data/{subID}/cluster'

		if exists(f'{subpath}/meanFC.{hemi}.K{Nclusters}.npy'):
			subconn = np.load(f'{subpath}/meanFC.{hemi}.K{Nclusters}.npy')
			meanconn[i, :, :] = subconn
			i = i + 1

	for roi in range(Nroi):
		roiconn = meanconn[:,:,roi]
		np.save(f'{store7}hblee/MPI/stat/roiFC/{hemi}-ROI{roi+1}.npy', roiconn)


'''
[ttest]
To perform two-sample t-test for each pair of data (FDR correction is applied)

Input:  /store7/hblee/MPI/stat/roiFC/{hemi}-ROI{i}.npy
Output: 1) /store7/hblee/MPI/stat/{hemi}-FC-t_statistics.npy
		2) /store7/hblee/MPI/stat/{hemi}-FC-t_pvalues.npy
'''
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
def ttest(hemi):
	Nroi = 82
	inpath = f'{store7}hblee/MPI/stat'

	t = np.zeros((Nroi, Nclusters))
	p = np.zeros((Nroi, Nclusters))
	corrected = np.zeros_like(p)
	for roi in range(Nroi):
		conn = np.load(f'{inpath}/roiFC/{hemi}-ROI{roi+1}.npy')
		t[roi,0], p[roi,0] = ttest_ind(conn[:,0], conn[:,1])
		t[roi,1], p[roi,1] = ttest_ind(conn[:,1], conn[:,2])
		t[roi,2], p[roi,2] = ttest_ind(conn[:,2], conn[:,0])

	for i in range(Nclusters):
		_, p_corr, _, _ = multipletests(p[:, i], 0.05, 'fdr_bh')
		corrected[:, i] = p_corr

	np.save(f'{inpath}/{hemi}-FC-t_statistics.npy', t)
	np.save(f'{inpath}/{hemi}-FC-t_pvalues.npy', corrected)
	
	df = pd.DataFrame(np.hstack((t, p, corrected)))
	filepath = if'{inpath}/{hemi}_stat_results.xlsx'
	df.to_excel(filepath, index=False)


'''
[main]
Main function to perform analysis
'''
def main(a, b, startname=None):
	datapath = store4 + 'hblee/4.MPI/4.clustFC/data/'
	sublist = listdir(datapath)
	if startname:
		a = sublist.index(startname)
	
	if b==0:
		sublist = sublist[a:]
	else:
		sublist = sublist[a:b]  # 162026 ~ 793465
	
	# 2. roiconn(hemi)
	# 3. ttest(hemi)
	for sidx, subID in enumerate(sublist):
		print('%dth sub - %s - FC analysis on cluster\n' %(sidx+1, subID))
		#1. compute_meanFC(subID)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="functional_connectivity")
	parser.add_argument(dest="startpoint",type=int,help="Start point of subject for data processing")
	parser.add_argument(dest="endpoint",type=int,help="End point of subject for data processing")
	parser.add_argument("-s",dest="startname",help="The name of the subject to start",required=False)
	args=parser.parse_args()
	main(args.startpoint, args.endpoint, args.verison, args.startname)
