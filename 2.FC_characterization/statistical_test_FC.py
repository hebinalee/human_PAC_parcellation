#################################################################
## TO CONDUCT STATISTICAL TEST
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

BASE_DIR = 'X:/path/myfolder'
DATA_DIR = BASE_DIR + '/data'

def set_subjdir(subID): return f'{DATA_DIR}/{subID}'

Nclusters = 3



'''
[roiconn]
To save matrix of meanFC values for each ROI (N_valid_subj X N_clusters)

Input:  {subj_dir}/cluster/relabel/meanFC.{hemi}.K3.npy (3 X 82)
Output: {BASE_DIR}/stat/roiFC/{hemi}-ROI{i}.npy
'''
def roiconn(hemi):
	sublist = sorted(listdir(DATA_DIR))
	Nsub = len(sublist)
	Nroi = 82

	meanconn = np.zeros((Nsub, Nclusters, Nroi))
	i = 0
	for sidx, subID in enumerate(sublist):
		subj_dir = set_subjdir(subID) + '/cluster'

		if exists(f'{subj_dir}/relabel/meanFC.{hemi}.K{Nclusters}.npy'):
			subconn = np.load(f'{subj_dir}/relabel/meanFC.{hemi}.K{Nclusters}.npy')
			meanconn[i, :, :] = subconn
			i = i + 1

	for roi in range(Nroi):
		roiconn = meanconn[:,:,roi]
		np.save(f'{BASE_DIR}/stat/roiFC/{hemi}-ROI{roi+1}.npy', roiconn)


'''
[ttest]
To perform two-sample t-test for each pair of data (FDR correction is applied)

Input:  {BASE_DIR}/stat/roiFC/{hemi}-ROI{i}.npy
Output: 1) {BASE_DIR}/stat/{hemi}-FC-t_statistics.npy
	2) {BASE_DIR}/stat/{hemi}-FC-t_pvalues.npy
'''
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
def ttest(hemi):
	Nroi = 82
	input_dir = f'{BASE_DIR}/stat'

	t = np.zeros((Nroi, Nclusters))
	p = np.zeros((Nroi, Nclusters))
	corrected = np.zeros_like(p)
	for roi in range(Nroi):
		conn = np.load(f'{input_dir}/roiFC/{hemi}-ROI{roi+1}.npy')
		t[roi,0], p[roi,0] = ttest_ind(conn[:,0], conn[:,1])
		t[roi,1], p[roi,1] = ttest_ind(conn[:,1], conn[:,2])
		t[roi,2], p[roi,2] = ttest_ind(conn[:,2], conn[:,0])

	for i in range(Nclusters):
		_, p_corr, _, _ = multipletests(p[:, i], 0.05, 'fdr_bh')
		corrected[:, i] = p_corr

	np.save(f'{input_dir}/{hemi}-FC-t_statistics.npy', t)
	np.save(f'{input_dir}/{hemi}-FC-t_pvalues.npy', corrected)
	
	df = pd.DataFrame(np.hstack((t, p, corrected)))
	filepath = if'{input_dir}/{hemi}_stat_results.xlsx'
	df.to_excel(filepath, index=False)


'''
[main]
Main function to perform analysis
'''
def main(a, b, startname=None):
	sublist = listdir(DATA_DIR)
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
