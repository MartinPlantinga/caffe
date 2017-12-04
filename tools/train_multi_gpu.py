#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Adaptation of $CAFFE_ROOT/python/train.py
# For training on multiple GPUs
# Modified by Martin Plantinga
# --------------------------------------------------------

"""
Trains a model using one or more GPUs.
"""
from multiprocessing import Process

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb

import caffe

import argparse
import pprint
import numpy as np
import sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpus', nargs='+', dest='gpu_ids',
                        help='GPU device id to use [0]',
                        default=[0], type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

def train_net_mgpu(
	solver,  # solver proto definition
	roidb,
	gpus,	# *listof device idsi
	output_dir,
	pretrained,
	snapshot,  # solver snapshot to restore
        max_iters,
	timing=False,  # show timing info for compute and communications
):
    # NCCL uses a uid to identify a session
    uid = caffe.NCCL.new_uid()

    caffe.init_log()
    caffe.log('Using devices %s' % str(gpus))

    procs = []
    for rank in range(len(gpus)):
        p = Process(target=solve,
                    args=(solver, 
			  snapshot, 
			  gpus, 
			  timing, 
			  uid, 
			  rank,
			  roidb,
			  max_iters,
			  output_dir,
			  pretrained))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


def time(solver, nccl):
    fprop = []
    bprop = []
    total = caffe.Timer()
    allrd = caffe.Timer()
    for _ in range(len(solver.net.layers)):
        fprop.append(caffe.Timer())
        bprop.append(caffe.Timer())
    display = solver.param.display

    def show_time():
        if solver.iter % display == 0:
            s = '\n'
            for i in range(len(solver.net.layers)):
                s += 'forw %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % fprop[i].ms
            for i in range(len(solver.net.layers) - 1, -1, -1):
                s += 'back %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % bprop[i].ms
            s += 'solver total: %.2f\n' % total.ms
            s += 'allreduce: %.2f\n' % allrd.ms
            caffe.log(s)

    solver.net.before_forward(lambda layer: fprop[layer].start())
    solver.net.after_forward(lambda layer: fprop[layer].stop())
    solver.net.before_backward(lambda layer: bprop[layer].start())
    solver.net.after_backward(lambda layer: bprop[layer].stop())
    solver.add_callback(lambda: total.start(), lambda: (total.stop(), allrd.start()))
    solver.add_callback(nccl)
    solver.add_callback(lambda: '', lambda: (allrd.stop(), show_time()))


def solve(proto, snapshot, gpus, timing, uid, rank, roidb, max_iters, output_dir, pretrained):
    caffe.set_mode_gpu()
    caffe.set_device(gpus[rank])
    caffe.set_solver_count(len(gpus))
    caffe.set_solver_rank(rank)
    caffe.set_multiprocess(True)

    nccl = caffe.NCCL(solver, uid)
    nccl.bcast()

    solver = caffe.SGDSolver(proto)
    train_net(	proto,
		roidb,
		output_dir,
		pretrained,
		max_iters)
		
    #if snapshot and len(snapshot) != 0:
    #    solver.restore(snapshot)

    #nccl = caffe.NCCL(solver, uid)
    #nccl.bcast()

    #if timing and rank == 0:
    #    time(solver, nccl)
    #else:
    #    solver.add_callback(nccl)

    #if solver.param.layer_wise_reduce:
    #    solver.net.after_backward(nccl)
    #solver.step(solver.param.max_iter)

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_ids

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    imdb, roidb = combined_roidb(args.imdb_name)
    print '{:d} roidb entries'.format(len(roidb))

    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    train_net_mgpu(	args.solver, 
			roidb, 
			args.gpu_ids,
			output_dir,
			args.pretrained_model,
			snapshot = None,
			max_iters=args.max_iters)

