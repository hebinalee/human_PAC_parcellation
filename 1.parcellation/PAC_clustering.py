#######################################
## Perform clustering of PAC using SC
#######################################
'''
[Order of function implementation]
1) clustering
2) compute_meanSC
3) relabel_clust
4) regist_clust
'''
import os
from os import listdir
from os.path import isfile, join, exists
import glob
import shutil
import numpy as np
from mayavi import mlab
import nibabel as nib
import matplotlib.pyplot as plt

BASE_DIR = 'X:/path/myfolder'
DATA_DIR = BASE_DIR + '/data'

def set_subjdir(subID): return f'{DATA_DIR}/{subID}'

Nclusters = 3

## seed ROI index:
## Left : 29(STG), 33(HG), 34(IS)
## Right: 78(STG), 82(HG), 83(IS)



'''
[map_t1w]
To save NIFTI file from labels and indices information

Input:  labels, indices, ref_anat
Output: out_path (volume image file)
'''
def map_t1w(labels, indices, ref_anat, out_path):
	import copy
	temp_ref = copy.deepcopy(ref_anat)
	temp_vol = np.zeros(temp_ref.shape)
	temp_vol[indices] = labels
	img = nib.Nifti1Image(temp_vol, temp_ref.affine)
	nib.save(img, out_path)


'''
[clustering]
To parcellate PAC by performing K-means clustering based on SC

Input:  1) {subj_dir}/tracto/fs_default.conn.matrix.npy
	2) {subj_dir}/tracto/fs_default.seed_idx.npy
Output: {subj_dir}/cluster/{hemi}.clust.K3.nii.gz
'''
# clustering4 : remove more regions with zero-std, include STG and IS
from sklearn.cluster import KMeans
def clustering(subID, hemi):
	subj_dir = set_input_dir(subID)
	conn_dir = join(subj_dir, 'tracto')
	clust_dir = join(subj_dir, 'cluster')
	if not os.path.exists(clust_dir):
		os.makedirs(clust_dir)

	# 1) Load input files (T1 image, connectivity matrix, seed indices)
	ref_anat = nib.load(join(subj_dir, 'T1w/T1w_acpc_dc_restore_brain.nii.gz'))
	# the connectivity map include null seed & null target voxels at first row & column
	conn_mat = np.load(join(conn_dir, 'fs_default.conn.matrix.npy'))[1:, 1:]
	seed_idx = np.load(join(conn_dir, 'fs_default.seed_idx.npy'))

	
	# 2) Consider voxels with connectivity value larger than 100
	valid_seed = np.where(conn_mat.sum(axis=1) >= 100)
	valid_conn = conn_mat[valid_seed]
	valid_conn = valid_conn / valid_conn.sum(axis=1).reshape(-1,1).astype(np.float64)
	valid_idx = np.transpose(seed_idx)[valid_seed]


	# 3) Divide data into Left/Right
	div_x = np.argmax(np.diff(valid_idx[:,0])) + 1	# the start point of x-axis in left hemisphere
	
	n_vox = len(valid_idx)
	hemi_set = np.array(div_x*['R'] + (n_vox-div_x)*['L'])

	# Remove target ROIs with low connectivity strength
	if hemi == 'L':
		target_roi = np.arange(42)
		ban_roi = np.array([5, 13, 25, 31, 32, 33, 42]) - 1
	elif hemi == 'R':
		target_roi = np.arange(42,84)
		ban_roi = np.array([49, 54, 52, 74, 80, 81, 82]) - 1

	valid_roi = np.setdiff1d(target_roi, ban_roi, assume_unique=True)
	hemi_conn = valid_conn[hemi_set==hemi][:,valid_roi]
	hemi_idx = valid_idx[hemi_set==hemi]


	# 4) Perform K-means clustering
	clf = KMeans(n_clusters = Nclusters, random_state = 0).fit(hemi_conn)
	orig_labels = clf.predict(hemi_conn)


	# 5) Save NIFTI file
	map_t1w(
			orig_labels + 1,
			tuple(hemi_idx.T),
			ref_anat,
			f'{clust_dir}/{hemi}.clust.K{.nii.gz' %(clust_dir, hemi, Nclusters)
			)


'''
[compute_meanSC]
To compute averaged SC values for each clusters

Input:  1) {subj_dir}/tracto/valid.conn.{hemi}.npy
	2) {subj_dir}/tracto/valid.idx.{hemi}.npy
	3) {subj_dir}/cluster/{hemi}.clust.K3.nii.gz
Output: {subj_dir}/cluster/meanSC.{hemi}.K3.npy
'''
def compute_meanSC(subID, hemi):
	subj_dir = set_input_dir(subID)
	out_dir = f'{BASE_DIR}/clustFC/figure/mean_sc/rawlabel'
	conn_dir = join(subj_dir, 'tracto')
	seed_conn = np.load(f'{conn_dir}/valid.conn.{hemi}.npy')
	seed_idx = np.load(f'{conn_dir}/valid.idx.{hemi}.npy')

	clust_dir = join(subj_dir, 'cluster')
	clust_file = f'{clust_dir}/{hemi}.clust.K{Nclusters}.nii.gz'
	clust = nib.load(clust_file).get_fdata()
	clust_label = np.array([clust[i[0], i[1], i[2]] for i in tuple(seed_idx)])

	averaged_sc = np.zeros((Nclusters, seed_conn.shape[1]))
	for label in range(1, Nclusters+1):
		idx = (clust_label == label)
		clust_sc = seed_conn[idx, :]
		averaged_sc[label-1, :] = np.mean(clust_sc, axis = 0)

	np.save(f'{clust_dir}/meanSC.{hemi}.K{Nclusters}.npy', averaged_sc)
	plt.imshow(averaged_sc)
	plt.savefig(f'{out_dir}/{hemi}-{subID}.png')


'''
[relabel]
To compute new label of clustering results based on meanSC values with STG ans IS

Input:  {subj_dir}/cluster/meanSC.{hemi}.K3.npy
Output: relabel
'''
def relabel(subID, hemi):
	subj_dir = set_input_dir(subID) + '/cluster'

	# 1) Get index for STG & IS
	if hemi == 'L':
		selected = np.array([27, 28])
	else:
		#selected = np.array([73, 74]) - 39
		selected = np.array([34, 35])
	relabel = np.zeros(Nclusters).astype(int)

	# 2) Load SC matrix
	sub_conn = np.load(f'{subj_dir}/meanSC.{hemi}.K{Nclusters}.npy')
	features = sub_conn[:, selected]

	# 3) Compute new label
	# Find cluster with the strongest SC with IS (Label -> 3)
	assigned = np.argmax(features[:,1])
	relabel[assigned] = 3
	# Find cluster with the strongest SC with STG among left 2 clusters (Label -> 1)
	rest = np.where(relabel==0)[0]
	relabel[rest[np.argmax(features[rest,0])]] = 1
	# The rest cluster (Label -> 2)
	relabel[relabel==0] = 2
	'''
	Swapped 1,2!
	Original code was wrong.
	It needed to be checked.
	'''
	return relabel


'''
[relabel_clust]
To align label of clustering results based on SC with STG ans IS and save NIFTI file
- relabel

Input:  1) {subj_dir}/cluster/{hemi}.clust.K3.nii.gz
	2) {subj_dir}/cluster/meanSC.{hemi}.K3.npy
Output: 1) {subj_dir}/cluster/relabel/{hemi}.clust.K3.relabel.nii.gz
	2) {subj_dir}/cluster/relabel/meanSC.{hemi}.K3.npy
'''
def relabel_clust(subID, hemi):
	subj_dir = set_input_dir(subID)
	clust_dir = join(subj_dir, 'cluster')
	out_dir = join(clust_dir, 'relabel')

	if hemi == 'L':
		refID = '174841'
	else:
		refID = '165638'
	
	if subID == refID:
		shutil.copy(f'{clust_dir}/{hemi}.clust.K{Nclusters}.nii.gz', f'{out_dir}/{hemi}.clust.K{Nclusters}.relabel.nii.gz')
	else:
		new_label = relabel(subID, hemi)
		
		if not exists(out_dir):
			os.makedirs(out_dir)

		# Relabel cluster label file (native volume space)
		ref_clust = nib.load(f'{clust_dir}/{hemi}.clust.K{Nclusters}.nii.gz')
		clust = ref_clust.get_fdata()
		relabeled = np.zeros_like(clust)
		#for label in range(Nclusters):
		#	relabeled[clust==(label+1)] = new_label[label]
		relabeled[clust==1]  = new_label[0]
		relabeled[clust==2]  = new_label[1]
		relabeled[clust==3]  = new_label[2]
		img = nib.Nifti1Image(relabeled, ref_clust.affine)
		nib.save(img, f'{out_dir}/{hemi}.clust.K{Nclusters}.relabel.nii.gz')

		# relabel meanSC file
		meansc = np.load(f'{clust_dir}/meanSC.{hemi}.K{Nclusters}.npy')
		sort_idx = np.argsort(new_label)[::-1]
		meansc = meansc[sort_idx, :]
		np.save(f'{out_dir}/meanSC.{hemi}.K{Nclusters}.npy', meansc)


'''
[regist_clust]
To register label data: native volume space -> standard volume space

Input:  {subj_dir}/cluster/relabel/{hemi}.clust.K3.relabel.nii.gz
Output: {subj_dir}/cluster/relabel/{hemi}.clust.K3.relabel.MNI2mm.nii.gz
'''
def regist_clust(subID):
	subj_dir = set_input_dir(subID)
	standard = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'
	warpfile = join(subj_dir, 'T1w/acpc_dc2standard.nii.gz')
	clust_dir = join(subj_dir, 'cluster')

	for hemi in ['L', 'R']:
		infile = clust_dir + '/relabel/%s.clust.K%d.relabel.nii.gz' %(hemi, Nclusters)
		outfile = clust_dir + '/relabel/%s.clust.K%d.relabel.MNI2mm.nii.gz' %(hemi, Nclusters)
		os.system('applywarp --rel --interp=nn -i %s -r %s -w %s -o %s' %(infile, standard, warpfile, outfile))


'''
[main]
Main function to perform analysis
'''
def main(a, b, startname=None):
	sublist = sorted(listdir(DATA_DIR))
	if startname:
		a = sublist.index(startname)
	
	if b==0:
		sublist = sublist[a:]
	else:
		sublist = sublist[a:b]  # 162026 ~ 793465

	for sidx, subID in enumerate(sublist):
		print(f'{sidx+1}th sub - {subID} - clustering\n')
		clustering(subID, 'L', Nclusters)
		clustering(subID, 'R', Nclusters)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="clustering")
	parser.add_argument(dest="startpoint",type=int,help="Start point of subject for data processing")
	parser.add_argument(dest="endpoint",type=int,help="End point of subject for data processing")
	parser.add_argument("-s",dest="startname",help="The name of the subject to start",required=False)
	args=parser.parse_args()
	main(args.startpoint, args.endpoint, args.startname)
