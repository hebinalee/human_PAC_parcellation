#################################################################
## STATISTICAL TEST
##
## roiconn : To save the FC matrix in target ROI-wise
## ttest   : To perform two-sample t-test for each pair of ROIs
##           and correct multiple comparison problem 
#################################################################
'''
[Order of function implementation]
1) roiconn
2) ttest
'''
import os
from os import listdir
from os.path import join, exists, isfile, isdir
import shutil
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
store7 = '/store7/'

Nclusters = 3


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
	
	# 1. roiconn(hemi)
	# 2. ttest(hemi)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="functional_connectivity")
	parser.add_argument(dest="startpoint",type=int,help="Start point of subject for data processing")
	parser.add_argument(dest="endpoint",type=int,help="End point of subject for data processing")
	parser.add_argument("-s",dest="startname",help="The name of the subject to start",required=False)
	args=parser.parse_args()
	main(args.startpoint, args.endpoint, args.verison, args.startname)
