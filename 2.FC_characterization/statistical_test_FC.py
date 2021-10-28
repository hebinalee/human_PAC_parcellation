#################################################################
## STATISTICAL TEST
##
## roiconn : To save the FC matrix in target ROI-wise
## ttest   : To perform two-sample t-test for each pair of ROIs
##           and correct multiple comparison problem 
#################################################################

import os
from os import listdir
from os.path import join, exists, isfile, isdir
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests
import pandas as pd

if os.name == 'nt':
	store4 = 'X:/'
else:
	store4 = '/store4/'


def roiconn(hemi, version, conntype):
	datapath = store4 + 'hblee/4.MPI/4.clustFC/data/'
	sublist = listdir(datapath)
	Nsub = len(sublist)
	Nroi = 41
	conn = ['SC', 'FC']
	connname = conn[conntype-1]

	ncluster = 3
	if version == 5:
		if hemi == 'L':
			ncluster = 4
		else:
			ncluster = 5
	meanconn = np.zeros((Nsub, ncluster, Nroi))
	i = 0
	for sidx, sname in enumerate(sublist):
		if version == 3:
			subpath = join(store4, 'hblee/4.MPI/4.clustFC/data', sname, 'cluster%d/relabel-SC4/' %version)
		else:
			subpath = join(store4, 'hblee/4.MPI/4.clustFC/data', sname, 'cluster%d/relabel-SC/' %version)

		if exists(subpath + 'mean%s.%s.K%d.npy' %(connname, hemi, ncluster)):
			subconn = np.load(subpath + 'mean%s.%s.K%d.npy' %(connname, hemi, ncluster))
			meanconn[i, :, :] = subconn
			i = i + 1

	for roi in range(Nroi):
		roiconn = meanconn[:,:,roi]
		np.save(store4 + 'hblee/4.MPI/4.clustFC/stat/cluster%d/roi%s/' %(version, connname) + '%s-ROI%d.npy' %(hemi, roi+1), roiconn)


def ttest(hemi, version, conntype):
	ncluster = 3
	Nroi = 41
	inpath = join(store4, 'hblee/4.MPI/4.clustFC/stat/cluster%d/' %version)
	#outpath = store4 + 'hblee/4.MPI/4.clustFC/stat/'
	conn = ['SC', 'FC']
	connname = conn[conntype-1]

	t = np.zeros((Nroi, ncluster))
	p = np.zeros((Nroi, ncluster))
	corrected = np.zeros_like(p)
	for roi in range(Nroi):
		conn = np.load(inpath + 'roi%s/%s-ROI%d.npy' %(connname, hemi, roi+1))
		t[roi,0], p[roi,0] = ttest_ind(conn[:,0], conn[:,1])
		t[roi,1], p[roi,1] = ttest_ind(conn[:,1], conn[:,2])
		t[roi,2], p[roi,2] = ttest_ind(conn[:,2], conn[:,0])

	for i in range(ncluster):
		_, p_corr, _, _ = multipletests(p[:, i], 0.05, 'fdr_bh')
		corrected[:, i] = p_corr

	np.save(inpath + '%s-%s-t_statistics.npy' %(hemi, connname), t)
	np.save(inpath + '%s-%s-t_pvalues.npy' %(hemi, connname), corrected)
	
	df = pd.DataFrame(np.hstack((t, p, corrected)))
	filepath = inpath + '%s_stat_results.xlsx' %hemi
	df.to_excel(filepath, index=False)
