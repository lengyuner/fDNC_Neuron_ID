"""
This file is used to synthesize worm neurons so that we can get more samples and potentially do registration on
id that is not labelled by old pipeline or arcoss worm.
"""
import matplotlib.pyplot as plt
import argparse
import os
import glob
import numpy as np
import pandas as pd
import torch
import tqdm
import pickle
import pickle
from Himag_cline import worm_cline, standard_neurons, find_min_match
from cpd_nonrigid_sep import register_nonrigid
#from cpd_plot import cpd_plot


def syn_head(neurons_s, labels, noise_var=0, rotate_yz=True, scale=200, label_mode='old', show=False,
             rotate_xy=True, affine=True, straighten=False):
    # synthesize head.
    pts_s_s = xyz
    # put z to 0
    pts_s_s[:, 2] -= np.median(pts_s_s[:, 2])

    pts_s_out = pts_s_s

    affine_lim = 0.2

    if rotate_yz:
        # rotate the cline in yoz plane.
        theta = (np.random.rand(1)[0]-0.5) * 0.9 * np.pi
        r_m = [[1, 0, 0], [0, np.cos(theta), np.sin(theta)], [0, -np.sin(theta), np.cos(theta)]]

        if affine:
            # Affine transform in x0y plane
            affine_m = np.array([[1, 0, 0], [0, 1, (np.random.rand(1)[0] - 0.5) * affine_lim],
                                 [0, (np.random.rand(1)[0] - 0.5) * affine_lim, 1]])
            r_m = np.matmul(r_m, affine_m)

        pts_s_out = np.matmul(pts_s_out, r_m)

    # add and miss some neurons.
    missing_prop = np.random.rand(1) * 0.2 + 0.05
    #missing_prop = 0.1
    num_neuron = pts_s_out.shape[0]
    add_lim = min(20, num_neuron * missing_prop)
    num_add = np.random.randint(add_lim, size=1)[0]
    num_add = max(1, num_add)
    max_x = np.max(pts_s_out[:,0])
    min_x = np.min(pts_s_out[:,0])
    max_y = np.max(pts_s_out[:,1])
    min_y = np.min(pts_s_out[:,1])
    max_z = np.max(pts_s_out[:,2])
    min_z = np.min(pts_s_out[:,2])

    pts_add = np.random.rand(num_add, 3) * np.array([[max_x-min_x, max_y-min_y, max_z-min_z]]) + np.array([[min_x, min_y, min_z]])
    _, dis_min = find_min_match(pts_add, pts_s_out)
    add_mask = np.where(dis_min > 5)[0]
    num_add = len(add_mask)
    pts_add = pts_add[add_mask, :]

    # add pts and label together.
    pts_s_out = np.vstack((pts_s_out, pts_add))

    label_ori = labels[:, np.newaxis]
    label = np.vstack((label_ori, np.ones((num_add, 1)) * -2))

    missing_rand = np.random.rand(len(pts_s_out))
    remain_idx = np.where(missing_rand > missing_prop)[0]
    pts_s_out = pts_s_out[remain_idx, :]
    label = label[remain_idx, :]

    if affine:
        if rotate_xy:
            theta = np.random.rand(1)[0] * 2 * np.pi
            r_m = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0],\
                   [0, 0, 1]])
        else:
            r_m = np.eye(3)

        scale_m = np.diag(np.random.rand(3) * 0.1 - 0.05) + np.eye(3)
        t_m = np.matmul(r_m, scale_m)
        #t_m = np.eye(3)
        affine_m = np.array([[1, (np.random.rand(1)[0] - 0.5) * affine_lim, 0], [(np.random.rand(1)[0] - 0.5) * affine_lim, 1, 0],
                             [0, 0, 1]])
        t_m = np.matmul(t_m, affine_m)

        pts_out = np.matmul(pts_s_out, t_m)

    if noise_var >0:
        pts_out += np.random.randn(pts_out.shape[0], pts_out.shape[1]) * noise_var

    pts_out -= np.median(pts_out, axis=0)
    out = np.hstack((pts_out/scale, label))

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Synthesize Neuron Data')
    parser.add_argument("--mode", type=str, default='multiple', help='synthesize same worm or across worm: copy, single or multiple')
    parser.add_argument("--index_mode", type=str, default='old', help='save old/new(0-num) track index')
    parser.add_argument("--path", type=str, default='/projects/LEIFER/Xinwei/github/NeuronNet/pts_id/real',
                        help='the path of real data(use your own real data(1 unit=0.42um)')
    parser.add_argument("--save_p", type=str, default='../results/train',
                        help='the path to save synthesized data')
    #parser.add_argument("--source_num", type=int, default=100, help='number of source worm selected randomly')
    #parser.add_argument("--template_num", type=int, default=64, help='number of template worm selected randomly')
    parser.add_argument("--batch_size", type=int, default=64, help="size of batch groups to generate")
    parser.add_argument("--batch_per_worm", type=int, default=40, help="number of batches to generate per worm")
    parser.add_argument("--shuffle", type=int, default=1, help="whether shuffle the dataset")
    parser.add_argument("--scale", type=float, default=84, help="the scale applied to original coordinates(1 unit=0.42um).")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--syn_mode", default="uns", type=str)
    parser.add_argument("--save_type", default='npy', type=str)
    parser.add_argument("--var", type=float, default=1,
                        help='variance add to points')
    #parser.add_argument()
    args = parser.parse_args()

    # load data in path
    aligned_worms = glob.glob(os.path.join(args.path, '*.csv'))
    #if args.shuffle:
    #    np.random.shuffle(aligned_worms)

    real_dict = dict()
    real_dict['files'] = aligned_worms
    real_dict['files_num'] = len(aligned_worms)

    atlas_file = open('/Users/danielysprague/foco_lab/data/atlases/2024_03_11_match_full_nosplit.pkl', 'rb')
    atlas = pickle.load(atlas_file)
    atlas_file.close

    atlas_neurons = np.asarray(atlas['names'])

    for file in tqdm.tqdm(real_dict['files']):

        filename = file.split('/')[-1]
        #print(file)
        
        if not os.path.exists(os.path.join(args.save_p, filename[:-4])):
            os.mkdir(os.path.join(args.save_p, filename[:-4]))

        blobs = pd.read_csv(file)
        xyz = np.asarray(blobs[['aligned_x', 'aligned_y', 'aligned_z']])
        rgb = np.asarray(blobs[['real_R', 'real_G', 'real_B']])
        IDs = np.asarray(blobs['ID'])

        labels = np.arange(len(IDs))

        for batch in range(args.batch_per_worm):
            if not os.path.exists(os.path.join(args.save_p, filename[:-4], filename[:-4]+'_batch'+str(batch))):
                os.mkdir(os.path.join(args.save_p, filename[:-4], filename[:-4]+'_batch'+str(batch)))
            for synth_worm in range(args.batch_size):

                pts_s_syn = syn_head(xyz, labels,label_mode= args.index_mode, scale= args.scale, noise_var=args.var)

                file_name = os.path.join(args.save_p, filename[:-4], filename[:-4]+'_batch'+str(batch), 'syn_{}_batch{}_{}.npy'.format(filename[:-4], str(batch), str(synth_worm)))
                np.save(file_name, pts_s_syn)

    # python src/data_syn_nocline.py --path /Users/danielysprague/foco_lab/data/aligned_2024_03_11/aligned_full --save_p Data/train_new 