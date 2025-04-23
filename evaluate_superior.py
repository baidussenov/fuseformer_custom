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
parser.add_argument("--neighbor_stride", type=int, default=5)
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
    i3d_model.load_state_dict(torch.load(i3d_model_weight))
    i3d_model.to(torch.device('cuda:0'))

def get_i3d_activations(batched_video, target_endpoint='Logits', flatten=True, grad_enabled=False):
    """
    Get features from i3d model and flatten them to 1d feature,
    valid target endpoints are defined in InceptionI3d.VALID_ENDPOINTS
    """
    init_i3d_model()
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
    print(f'Reading frames from video {vname}...')
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


import subprocess as sp

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def main_worker():
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('model.' + args.model)
    print('GPU check 1', get_gpu_memory())
    model = net.InpaintGenerator().to(device)
    print('GPU check 2', get_gpu_memory())
    model_path = args.ckpt
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data)
    print('loading from: {}'.format(args.ckpt))
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
            
    for video_no in range(video_num):
        print("[Processing: {}]".format(frame_list[video_no].split("/")[-1]))
        if args.dump_results:
            this_dump_results_dir = os.path.join(dump_results_dir, frame_list[video_no].split("/")[-1])
            os.makedirs(this_dump_results_dir, exist_ok=True)

        # Get video length without loading all frames
        frame_from_videos = read_frame_from_videos(frame_list[video_no])
        video_length = len(frame_from_videos)
        comp_frames = [None] * video_length
        gt_frames = [None] * video_length  # Store ground truth frames

        # Define chunking parameters
        max_frames_per_chunk = 30
        num_chunks = (video_length + max_frames_per_chunk - 1) // max_frames_per_chunk

        # Process video in chunks
        for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks", leave=False):
            # Compute frame indices for this chunk
            start_idx = chunk_idx * max_frames_per_chunk
            end_idx = min(start_idx + max_frames_per_chunk, video_length)
            chunk_indices = list(range(start_idx, end_idx))
            chunk_size = len(chunk_indices)
            # print(f"Chunk {chunk_idx}: indices {chunk_indices}, size {chunk_size}")

            # Load frames for this chunk
            frames_PIL = [frame_from_videos[i] for i in chunk_indices]
            # print(f"Frame size: {frames_PIL[0].size}")
            imgs = _to_tensors(frames_PIL).unsqueeze(0) * 2 - 1  # Shape: (1, chunk_size, 3, h, w)
            frames = [np.array(f).astype(np.uint8) for f in frames_PIL]

            # Store frames in gt_frames
            for i, frame_idx in enumerate(chunk_indices):
                gt_frames[frame_idx] = frames[i]

            # Load and replicate mask for this chunk
            single_mask = Image.fromarray(np.array(read_single_mask(mask_path, 1)[0]))
            masks = [single_mask] * chunk_size
            binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
            masks = _to_tensors(masks).unsqueeze(0)  # Shape: (1, chunk_size, 1, h, w)

            # Verify shapes
            # print(f"Imgs shape: {imgs.shape}, Masks shape: {masks.shape}")

            # Move to GPU
            # print('GPU check 3', get_gpu_memory())
            selected_imgs = imgs.to(device)
            selected_masks = masks.to(device)
            # print('GPU check 4', get_gpu_memory())

            with torch.no_grad():
                from torch.amp import autocast
                with autocast('cuda'):
                    input_imgs = selected_imgs * (1 - selected_masks)
                    # print('Input IMGs shape:', input_imgs.shape)
                    pred_img = model(input_imgs)
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255  # Shape: (chunk_size, h, w, 3)

            # Update comp_frames
            for i, frame_idx in enumerate(chunk_indices):
                img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] + frames[i] * (1 - binary_masks[i])
                if comp_frames[frame_idx] is None:
                    comp_frames[frame_idx] = img
                else:
                    comp_frames[frame_idx] = comp_frames[frame_idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5

            # Clear GPU and CPU memory
            del selected_imgs, selected_masks, input_imgs, pred_img, imgs, masks, frames_PIL, frames, binary_masks
            torch.cuda.empty_cache()
            # print('GPU check 5', get_gpu_memory())

        # Evaluation loop
        ssim, psnr, s_psnr = 0., 0., 0.
        comp_PIL = []
        for f in range(video_length):
            comp = comp_frames[f]
            comp = cv2.cvtColor(np.array(comp), cv2.COLOR_BGR2RGB)

            cv2.imwrite("tmpp.png", comp)
            new_comp = cv2.imread("tmpp.png")
            new_comp = Image.fromarray(cv2.cvtColor(new_comp, cv2.COLOR_BGR2RGB))
            comp_PIL.append(new_comp)

            if args.dump_results:
                cv2.imwrite(os.path.join(this_dump_results_dir, "{:04}.png".format(f)), comp)

            # Use gt_frames for ground truth
            gt = cv2.cvtColor(np.array(gt_frames[f]).astype(np.uint8), cv2.COLOR_BGR2RGB)
            ssim += ssim_func(comp, gt, data_range=255, channel_axis=-1, win_size=65)
            s_psnr += psnr_func(gt, comp, data_range=255)

        ssim_all += ssim
        s_psnr_all += s_psnr
        video_length_all += video_length
        if video_no % 5 == 0:
            print("ssim {}, psnr {}".format(ssim_all / video_length_all, s_psnr_all / video_length_all))

        # Clear CPU memory
        del comp_frames, gt_frames, frame_from_videos
        torch.cuda.empty_cache()

        # FVID computation (commented out as in original)
        # imgs = _to_tensors(comp_PIL).unsqueeze(0).to(device)
        # gts = _to_tensors([Image.fromarray(gt_frames[f]) for f in range(video_length)]).unsqueeze(0).to(device)
        # output_i3d_activations.append(get_i3d_activations(imgs).cpu().numpy().flatten())
        # real_i3d_activations.append(get_i3d_activations(gts).cpu().numpy().flatten())
        
    fid_score = get_fid_score(real_i3d_activations, output_i3d_activations)
    print("[Finish evaluating, ssim is {}, psnr is {}]".format(ssim_all / video_length_all, s_psnr_all / video_length_all))
    print("[fvid score is {}]".format(fid_score))

# def main_worker():
#     # set up models 
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net = importlib.import_module('model.' + args.model)
#     model = net.InpaintGenerator().to(device)
#     model_path = args.ckpt
#     data = torch.load(args.ckpt, map_location=device)
#     model.load_state_dict(data)
#     print('loading from: {}'.format(args.ckpt))
#     model.eval()

#     frame_list, mask_path = get_frame_list(args)
#     print('Num of frames: ', len(frame_list))
#     video_num = len(frame_list)

#     ssim_all, psnr_all, len_all = 0., 0., 0.
#     s_psnr_all = 0.
#     video_length_all = 0
#     vfid = 0.
#     output_i3d_activations = []
#     real_i3d_activations = []

#     model_name = args.ckpt.split("/")[-1].split(".")[0]
#     dump_results_dir = model_name+"_single_mask_results"
#     if args.dump_results:
#         if not os.path.exists(dump_results_dir):
#             os.mkdir(dump_results_dir)
            
#     # Add tqdm for video processing loop
#     for video_no in tqdm(range(video_num), desc="Processing videos"):
#         print("[Processing: {}]".format(frame_list[video_no].split("/")[-1]))
#         if args.dump_results:
#             this_dump_results_dir = os.path.join(dump_results_dir, frame_list[video_no].split("/")[-1])
#             os.makedirs(this_dump_results_dir, exist_ok=True)

#         frames_PIL = read_frame_from_videos(frame_list[video_no])
#         video_length = len(frames_PIL)
#         masks_PIL = read_single_mask(mask_path, video_length)

#         # Don't convert to tensors and move to GPU here
#         frames = [np.array(f).astype(np.uint8) for f in frames_PIL]
#         binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks_PIL]
#         comp_frames = [None]*video_length

#         # Process in chunks with tqdm
#         chunk_size = 30  # Adjust based on your GPU memory
#         chunk_pbar = tqdm(range(0, video_length, chunk_size), desc="Processing chunks", leave=False)
#         for chunk_start in chunk_pbar:
#             chunk_end = min(chunk_start + chunk_size, video_length)
#             chunk_frames = frames_PIL[chunk_start:chunk_end]
#             chunk_masks = masks_PIL[chunk_start:chunk_end]
            
#             # Update chunk progress bar with info
#             chunk_pbar.set_postfix({
#                 "chunk": f"{chunk_start}-{chunk_end}",
#                 "frames": len(chunk_frames)
#             })
            
#             # Convert only this chunk to tensors and move to GPU
#             chunk_imgs = _to_tensors(chunk_frames).unsqueeze(0)*2-1
#             chunk_masks_tensor = _to_tensors(chunk_masks).unsqueeze(0)
#             chunk_imgs, chunk_masks_tensor = chunk_imgs.to(device), chunk_masks_tensor.to(device)
            
#             # Process frames within this chunk
#             frame_pbar = tqdm(range(chunk_start, chunk_end, neighbor_stride), 
#                              desc="Processing frames", leave=False)
#             for f in frame_pbar:
#                 local_f = f - chunk_start  # Local index within the chunk
                
#                 # Calculate neighbor frames within the chunk
#                 neighbor_ids = [i for i in range(
#                     max(0, local_f-neighbor_stride), 
#                     min(chunk_end-chunk_start, local_f+neighbor_stride+1)
#                 )]
                
#                 # For reference frames, you might need to load them separately
#                 global_neighbor_ids = [n + chunk_start for n in neighbor_ids]
#                 ref_ids = get_ref_index(f, global_neighbor_ids, video_length)
                
#                 # Show info in frame progress bar
#                 frame_pbar.set_postfix({
#                     "frame": f,
#                     "refs": len(ref_ids)
#                 })
                
#                 # Handle reference frames from outside the current chunk
#                 extra_ref_frames = []
#                 extra_ref_masks = []
#                 in_chunk_ref_ids = []
                
#                 for ref_id in ref_ids:
#                     if chunk_start <= ref_id < chunk_end:
#                         # Reference frame is within the current chunk
#                         in_chunk_ref_ids.append(ref_id - chunk_start)
#                     else:
#                         # Reference frame is outside the current chunk
#                         extra_ref_frames.append(frames_PIL[ref_id])
#                         extra_ref_masks.append(masks_PIL[ref_id])
                
#                 if extra_ref_frames:
#                     extra_imgs = _to_tensors(extra_ref_frames).unsqueeze(0)*2-1
#                     extra_masks = _to_tensors(extra_ref_masks).unsqueeze(0)
#                     extra_imgs, extra_masks = extra_imgs.to(device), extra_masks.to(device)
                    
#                     # Combine with chunk tensors for processing
#                     if in_chunk_ref_ids:
#                         selected_imgs = torch.cat([
#                             chunk_imgs[:1, neighbor_ids, :, :, :],
#                             chunk_imgs[:1, in_chunk_ref_ids, :, :, :],
#                             extra_imgs
#                         ], dim=1)
#                         selected_masks = torch.cat([
#                             chunk_masks_tensor[:1, neighbor_ids, :, :, :],
#                             chunk_masks_tensor[:1, in_chunk_ref_ids, :, :, :],
#                             extra_masks
#                         ], dim=1)
#                     else:
#                         selected_imgs = torch.cat([
#                             chunk_imgs[:1, neighbor_ids, :, :, :], 
#                             extra_imgs
#                         ], dim=1)
#                         selected_masks = torch.cat([
#                             chunk_masks_tensor[:1, neighbor_ids, :, :, :], 
#                             extra_masks
#                         ], dim=1)
#                 else:
#                     # Use only frames from the current chunk
#                     if in_chunk_ref_ids:
#                         all_ids = neighbor_ids + in_chunk_ref_ids
#                         selected_imgs = chunk_imgs[:1, all_ids, :, :, :]
#                         selected_masks = chunk_masks_tensor[:1, all_ids, :, :, :]
#                     else:
#                         selected_imgs = chunk_imgs[:1, neighbor_ids, :, :, :]
#                         selected_masks = chunk_masks_tensor[:1, neighbor_ids, :, :, :]
                
#                 with torch.no_grad():
#                     input_imgs = selected_imgs*(1-selected_masks)
#                     pred_img = model(input_imgs)
#                     pred_img = (pred_img + 1) / 2
#                     pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy()*255
                    
#                     # Update the appropriate frames
#                     for i, local_idx in enumerate(neighbor_ids):
#                         global_idx = local_idx + chunk_start
#                         img = np.array(pred_img[i]).astype(np.uint8)*binary_masks[global_idx] + \
#                               frames[global_idx] * (1-binary_masks[global_idx])
#                         if comp_frames[global_idx] is None:
#                             comp_frames[global_idx] = img
#                         else:
#                             comp_frames[global_idx] = comp_frames[global_idx].astype(np.float32)*0.5 + \
#                                                     img.astype(np.float32)*0.5
                
#                 # Clear GPU cache after processing each frame if needed
#                 if video_length > 200:  # Only for very large videos
#                     torch.cuda.empty_cache()
            
#             # Clear these tensors explicitly before the next chunk
#             del chunk_imgs, chunk_masks_tensor
#             if 'extra_imgs' in locals():
#                 del extra_imgs, extra_masks
#             torch.cuda.empty_cache()

#         # Continue with your existing metrics calculation
#         ssim, psnr, s_psnr = 0., 0., 0.
#         comp_PIL = []
#         for f in range(video_length):
#             comp = comp_frames[f]
#             comp = cv2.cvtColor(np.array(comp), cv2.COLOR_BGR2RGB)

#             cv2.imwrite("tmpp.png", comp)
#             new_comp = cv2.imread("tmpp.png")
#             new_comp = Image.fromarray(cv2.cvtColor(new_comp, cv2.COLOR_BGR2RGB))
#             comp_PIL.append(new_comp)

#             if args.dump_results:
#                 cv2.imwrite(os.path.join(this_dump_results_dir, "{:04}.png".format(f)), comp)
#             gt = cv2.cvtColor(np.array(frames[f]).astype(np.uint8), cv2.COLOR_BGR2RGB)
#             ssim += ssim_func(comp, gt, data_range=255, channel_axis=-1, win_size=65)
#             s_psnr += psnr_func(gt, comp, data_range=255)

#         ssim_all += ssim
#         s_psnr_all += s_psnr
#         video_length_all += (video_length)
#         if video_no % 5 == 0:
#             print("ssim {}, psnr {}".format(ssim_all/video_length_all, s_psnr_all/video_length_all))
            
#         # FVID computation - also process in chunks to avoid memory issues
#         fvid_chunk_size = 10  # Process 10 frames at a time for FID
#         output_activations = []
#         real_activations = []
        
#         for i in tqdm(range(0, video_length, fvid_chunk_size), desc="Computing FID", leave=False):
#             end_i = min(i + fvid_chunk_size, video_length)
            
#             # Process output frames
#             out_imgs = _to_tensors(comp_PIL[i:end_i]).unsqueeze(0).to(device)
#             out_activations = get_i3d_activations(out_imgs).cpu().numpy()
#             output_activations.append(out_activations)
            
#             # Process ground truth frames
#             gt_imgs = _to_tensors(frames_PIL[i:end_i]).unsqueeze(0).to(device)
#             gt_activations = get_i3d_activations(gt_imgs).cpu().numpy()
#             real_activations.append(gt_activations)
            
#             # Clear memory
#             del out_imgs, gt_imgs
#             torch.cuda.empty_cache()
        
#         # Combine and flatten activations
#         output_i3d_activations.append(np.vstack(output_activations).flatten())
#         real_i3d_activations.append(np.vstack(real_activations).flatten())
        
#     fid_score = get_fid_score(real_i3d_activations, output_i3d_activations)
#     print("[Finish evaluating, ssim is {}, psnr is {}]".format(ssim_all/video_length_all, s_psnr_all/video_length_all))
#     print("[fvid score is {}]".format(fid_score))

if __name__ == '__main__':
    main_worker()