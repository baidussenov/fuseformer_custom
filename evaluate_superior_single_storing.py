import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import importlib
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
import copy
import random
import sys
import json
from skimage import measure
from skimage.metrics import structural_similarity as ssim_func, peak_signal_noise_ratio as psnr_func
from core.utils import create_random_shape_with_random_motion

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.multiprocessing as mp
from torchvision import transforms

# My libs
from core.utils import Stack, ToTorchFormatTensor
from model.i3d import InceptionI3d
from scipy import linalg
from tqdm import tqdm


parser = argparse.ArgumentParser(description="FuseFormer")
parser.add_argument("-v", "--video", type=str, required=False)
parser.add_argument("-c", "--ckpt",   type=str, required=True)
parser.add_argument("--model", type=str, default='fuseformer')
parser.add_argument("--dataset", type=str, default='davis')
parser.add_argument("--width", type=int, default=432)
parser.add_argument("--height", type=int, default=240)
parser.add_argument("--outw", type=int, default=432)
parser.add_argument("--outh", type=int, default=240)
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--num_ref", type=int, default=-1)
parser.add_argument("--neighbor_stride", type=int, default=1)
parser.add_argument("--savefps", type=int, default=24)
parser.add_argument("--use_mp4", action='store_true')
parser.add_argument("--dump_results", action='store_true')
args = parser.parse_args()


w, h = args.width, args.height
ref_length = args.step  # ref_step
num_ref = args.num_ref
neighbor_stride = args.neighbor_stride
default_fps = args.savefps
i3d_model = None

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +  # NOQA
            np.trace(sigma2) - 2 * tr_covmean)

def get_fid_score(real_activations, fake_activations):
    """
    Given two distribution of features, compute the FID score between them
    """
    m1 = np.mean(real_activations, axis=0)
    m2 = np.mean(fake_activations, axis=0)
    s1 = np.cov(real_activations, bias=1)
    s2 = np.cov(fake_activations, bias=1)

    return calculate_frechet_distance(m1, s1, m2, s2)

def init_i3d_model():
    global i3d_model
    if i3d_model is not None:
        return

    print("[Loading I3D model for FID score ..]")
    i3d_model_weight = './checkpoints/i3d_rgb_imagenet.pt'
    i3d_model = InceptionI3d(400, in_channels=3, final_endpoint='Logits')
    i3d_model.load_state_dict(torch.load(i3d_model_weight, map_location='cpu'))
    i3d_model.to(torch.device('cpu'))

# def get_i3d_activations(batched_video, target_endpoint='Logits', flatten=True, grad_enabled=False):
#     """
#     Get features from i3d model and flatten them to 1d feature,
#     valid target endpoints are defined in InceptionI3d.VALID_ENDPOINTS
#     """
#     init_i3d_model()
#     with torch.set_grad_enabled(grad_enabled):
#         feat = i3d_model.extract_features(batched_video.transpose(1, 2), target_endpoint)
#     if flatten:
#         feat = feat.view(feat.size(0), -1)

#     return feat

def get_i3d_activations(batched_video, target_endpoint='Logits', flatten=True, grad_enabled=False):
    """
    Get features from i3d model and flatten them to 1d feature,
    valid target endpoints are defined in InceptionI3d.VALID_ENDPOINTS
    """
    init_i3d_model()
    device = next(i3d_model.parameters()).device  # Get the device of the model
    
    # Make sure the input is on the same device as the model
    if batched_video.device != device:
        batched_video = batched_video.to(device)
        
    with torch.set_grad_enabled(grad_enabled):
        feat = i3d_model.extract_features(batched_video.transpose(1, 2), target_endpoint)
    if flatten:
        feat = feat.view(feat.size(0), -1)

    return feat

def get_frame_list(args):
    if args.dataset == 'davis':
        data_root = "./data/DATASET_DAVIS"
        frame_dir = os.path.join(data_root, "JPEGImages", "480p")
    elif args.dataset == 'youtubevos':
        data_root = "./data/YouTubeVOS/"
        frame_dir = os.path.join(data_root, "test_all_frames", "JPEGImages")
    else:
        frame_dir = "dataset_copy/normal_light_10"
        mask_path = '../masker/output_masks/S01_colour_chart/masks/00001.png'

    frame_folder = sorted(os.listdir(frame_dir))
    frame_list = [os.path.join(frame_dir, name) for name in frame_folder]

    print("[Finish building dataset {}]".format(args.dataset))
    return frame_list, mask_path

# sample reference frames from the whole video 
def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if not i in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref//2))
        end_idx = min(length, f + ref_length * (num_ref//2))
        for i in range(start_idx, end_idx+1, ref_length):
            if not i in neighbor_ids:
                ref_index.append(i)
                if len(ref_index) >= num_ref:
                    break
    return ref_index

# Read a single mask and replicate it for all frames
def read_single_mask(mask_path, video_length):
    # Read the single mask
    mask = Image.open(mask_path)
    mask = mask.resize((w, h), Image.NEAREST)
    mask = np.array(mask.convert('L'))
    mask = np.array(mask > 0).astype(np.uint8)
    mask = cv2.dilate(mask, cv2.getStructuringElement(
        cv2.MORPH_CROSS, (3, 3)), iterations=4)
    
    # Convert to PIL Image
    mask_pil = Image.fromarray(mask*255)
    
    # Replicate for all frames
    masks = [mask_pil] * video_length
    return masks

#  read frames from video 
def read_frame_from_videos(vname):
    lst = os.listdir(vname)
    lst.sort()
    fr_lst = [vname+'/'+name for name in lst]
    frames = []
    for fr in fr_lst:
        image = cv2.imread(fr)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frames.append(image.resize((w,h)))
    return frames

def get_res_list(dir):
    folders = sorted(os.listdir(dir))
    return [os.path.join(dir, f) for f in folders]

def main_worker():
    # Set up models 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator().to(device)
    model_path = args.ckpt
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data)
    print('Loading from: {}'.format(args.ckpt))
    model.eval()

    frame_list, mask_path = get_frame_list(args)
    print('Num of frames: ', len(frame_list))
    video_num = len(frame_list)

    ssim_all, psnr_all, len_all = 0., 0., 0.
    s_psnr_all = 0.
    video_length_all = 0
    vfid = 0.
    output_i3d_activations = []
    real_i3d_activations = []

    model_name = args.ckpt.split("/")[-1].split(".")[0]
    dump_results_dir = model_name + "_single_mask_results"
    if args.dump_results:
        if not os.path.exists(dump_results_dir):
            os.mkdir(dump_results_dir)
            
    # Add tqdm for video processing loop
    for video_no in tqdm(range(video_num), desc="Processing videos"):
        print("[Processing: {}]".format(frame_list[video_no].split("/")[-1]))
        if args.dump_results:
            this_dump_results_dir = os.path.join(dump_results_dir, frame_list[video_no].split("/")[-1])
            os.makedirs(this_dump_results_dir, exist_ok=True)

        frames_PIL = read_frame_from_videos(frame_list[video_no])
        video_length = len(frames_PIL)
        imgs = _to_tensors(frames_PIL).unsqueeze(0) * 2 - 1
        frames = [np.array(f).astype(np.uint8) for f in frames_PIL]

        # Read the single mask and replicate it for all frames
        masks = read_single_mask(mask_path, video_length)
        binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
        masks = _to_tensors(masks).unsqueeze(0)
    
        imgs, masks = imgs.to(device), masks.to(device)

        # Add tqdm for frame processing loop
        for f in tqdm(range(0, video_length, neighbor_stride), desc="Processing frames", leave=False):
            neighbor_ids = [i for i in range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1))]
            ref_ids = get_ref_index(f, neighbor_ids, video_length)
            len_temp = len(neighbor_ids) + len(ref_ids)
            selected_imgs = imgs[:1, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]
            
            with torch.no_grad():
                input_imgs = selected_imgs * (1 - selected_masks)
                pred_img = model(input_imgs)
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[idx] + frames[idx] * (1 - binary_masks[idx])
                    
                    # Save the result immediately
                    if args.dump_results:
                        output_path = os.path.join(this_dump_results_dir, "{:04}.png".format(idx))
                        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    
                    # Calculate SSIM and PSNR
                    gt = cv2.cvtColor(np.array(frames[idx]).astype(np.uint8), cv2.COLOR_BGR2RGB)
                    ssim_all += ssim_func(img, gt, data_range=255, channel_axis=-1, win_size=65)
                    s_psnr_all += psnr_func(gt, img, data_range=255)

        # Clear GPU memory after processing each video
        del imgs, masks, selected_imgs, selected_masks, input_imgs, pred_img
        torch.cuda.empty_cache()

        # FVID computation
        comp_PIL = [Image.open(os.path.join(this_dump_results_dir, "{:04}.png".format(f))) for f in range(video_length)]
        # imgs = _to_tensors(comp_PIL).unsqueeze(0).to(device)
        # gts = _to_tensors(frames_PIL).unsqueeze(0).to(device)
        # output_i3d_activations.append(get_i3d_activations(imgs).cpu().numpy().flatten())
        # real_i3d_activations.append(get_i3d_activations(gts).cpu().numpy().flatten())

        imgs = _to_tensors(comp_PIL).unsqueeze(0).to('cpu')  # Move to CPU
        gts = _to_tensors(frames_PIL).unsqueeze(0).to('cpu')  # Move to CPU
        output_i3d_activations.append(get_i3d_activations(imgs).numpy().flatten())
        real_i3d_activations.append(get_i3d_activations(gts).numpy().flatten())

        # Clear GPU memory again
        del imgs, gts
        torch.cuda.empty_cache()

        video_length_all += video_length
        if video_no % 5 == 0:
            print("SSIM {}, PSNR {}".format(ssim_all / video_length_all, s_psnr_all / video_length_all))

    # Calculate FID score
    fid_score = get_fid_score(real_i3d_activations, output_i3d_activations)
    print("[Finish evaluating, SSIM is {}, PSNR is {}]".format(ssim_all / video_length_all, s_psnr_all / video_length_all))
    print("[FVID score is {}]".format(fid_score))
    
if __name__ == '__main__':
    main_worker()