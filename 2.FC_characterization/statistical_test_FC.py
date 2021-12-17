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

basepath = 'X:/path/myfolder/'
Nclusters = 3

def roiconn(hemi, conntype):
	datapath = basepath + 'data/'
	sublist = listdir(datapath)
	Nsub = len(sublist)
	Nroi = 41
	conn = ['SC', 'FC']
	connname = conn[conntype-1]

	meanconn = np.zeros((Nsub, Nclusters, Nroi))
	i = 0
	for sidx, sname in enumerate(sublist):
		subpath = join(basepath, 'data', sname, 'cluster/relabel-SC/')

		if exists(subpath + 'mean%s.%s.K%d.npy' %(connname, hemi, Nclusters)):
			subconn = np.load(subpath + 'mean%s.%s.K%d.npy' %(connname, hemi, Nclusters))
			meanconn[i, :, :] = subconn
			i = i + 1

	for roi in range(Nroi):
		roiconn = meanconn[:,:,roi]
		np.save(f'{basepath}/stat/cluster/roi{connname}/{hemi}-ROI{roi+1}.npy', roiconn)


def ttest(hemi, conntype):
	Nroi = 41
	inpath = join(basepath, 'stat/cluster')
	conn = ['SC', 'FC']
	connname = conn[conntype-1]

	t = np.zeros((Nroi, Nclusters))
	p = np.zeros((Nroi, Nclusters))
	corrected = np.zeros_like(p)
	for roi in range(Nroi):
		conn = np.load(f'{inpath}/roi{connname}/{hemi}-ROI{roi+1}.npy')
		t[roi,0], p[roi,0] = ttest_ind(conn[:,0], conn[:,1])
		t[roi,1], p[roi,1] = ttest_ind(conn[:,1], conn[:,2])
		t[roi,2], p[roi,2] = ttest_ind(conn[:,2], conn[:,0])

	for i in range(Nclusters):
		_, p_corr, _, _ = multipletests(p[:, i], 0.05, 'fdr_bh')
		corrected[:, i] = p_corr

	np.save(f'{inpath}/{hemi}-{connname}-t_statistics.npy', t)
	np.save(f'{inpath}/{hemi}-{connname}-t_pvalues.npy', corrected)
	
	df = pd.DataFrame(np.hstack((t, p, corrected)))
	filepath = inpath + '%s_stat_results.xlsx' %hemi
	df.to_excel(filepath, index=False)
