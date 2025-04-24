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
import lpips

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
parser.add_argument("--neighbor_stride", type=int, default=10)
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

def init_lpips_model():
    """Initialize the LPIPS model for perceptual similarity measurements"""
    global lpips_model
    if lpips_model is not None:
        return
        
    print("[Loading LPIPS model for perceptual similarity ..]")
    lpips_model = lpips.LPIPS(net='alex').to(torch.device('cuda:0'))  # Use AlexNet for LPIPS
    lpips_model.eval()

def compute_lpips(img0, img1):
    """
    Compute LPIPS perceptual similarity between two images
    Input: img0, img1 as torch tensors in range [-1, 1] (BCHW)
    Returns: similarity distance (lower is more similar)
    """
    init_lpips_model()
    with torch.no_grad():
        return lpips_model(img0, img1).item()

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
        frame_dir = "dataset_copy/low_light_10"
        masks_path = '../masker/output_masks'

    frame_folder = sorted(os.listdir(frame_dir))
    frame_list = [os.path.join(frame_dir, name) for name in frame_folder]

    print("[Finish building dataset {}]".format(args.dataset))
    return frame_list, masks_path

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

def read_single_mask(mask_path, video_length=None):
    """
    Read and process a single mask image for a specific frame.
    
    Args:
        mask_path (str): Path to the mask image file (e.g., '00001.png').
        video_length (int, optional): Not used, included for compatibility.
    
    Returns:
        PIL.Image: Processed mask as a PIL Image.
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    mask = Image.open(mask_path)
    mask = mask.resize((w, h), Image.NEAREST)  # Ensure w, h are defined
    mask = np.array(mask.convert('L'))  # Convert to grayscale
    mask = np.array(mask > 0).astype(np.uint8)  # Binarize
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)
    mask_pil = Image.fromarray(mask * 255)  # Convert to PIL Image
    return mask_pil

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

def main_worker():
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator().to(device)
    model_path = args.ckpt
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data)
    print('loading from: {}'.format(args.ckpt))

    # Extract the state_dict based on checkpoint structure
    if 'netG' in data:
        state_dict = data['netG']
        print("Extracted 'netG' state_dict from checkpoint")
    else:
        state_dict = data
        print("Using direct state_dict from checkpoint")

    model.load_state_dict(state_dict)
    print('Loaded model from: {}'.format(args.ckpt))
    model.eval()

    frame_list, masks_path = get_frame_list(args)
    print('Num of frames: ', len(frame_list))
    video_num = len(frame_list)

    ssim_all, psnr_all, len_all = 0., 0., 0.
    s_psnr_all = 0.
    lpips_all = 0.
    video_length_all = 0
    vfid = 0.
    output_i3d_activations = []
    real_i3d_activations = []

    model_name = args.ckpt.split("/")[-1].split(".")[0]
    dump_results_dir = model_name + "_single_mask_results_low10"
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
        all_comp_PIL = []  # Store all completed frames for VFID
        all_gt_PIL = []    # Store all ground truth frames for VFID

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

            # Load frames for this chunk
            frames_PIL = [frame_from_videos[i] for i in chunk_indices]
            imgs = _to_tensors(frames_PIL).unsqueeze(0) * 2 - 1  # Shape: (1, chunk_size, 3, h, w)
            frames = [np.array(f).astype(np.uint8) for f in frames_PIL]

            # Store frames in gt_frames
            for i, frame_idx in enumerate(chunk_indices):
                gt_frames[frame_idx] = frames[i]

            # Load masks for this chunk
            selected_masks = []
            for frame_idx in chunk_indices:
                mask_path = os.path.join(masks_path, os.path.basename(frame_list[video_no]), 'masks', f'{frame_idx + 1:05d}.png')
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"Mask file not found: {mask_path}")
                mask_pil = read_single_mask(mask_path)
                selected_masks.append(mask_pil)

            # Convert masks to tensors
            binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in selected_masks]
            masks_tensor = _to_tensors(selected_masks).unsqueeze(0)  # Shape: (1, chunk_size, 1, h, w)

            # Move to GPU
            selected_imgs = imgs.to(device)
            selected_masks_tensor = masks_tensor.to(device)

            with torch.no_grad():
                from torch.amp import autocast
                with autocast('cuda'):
                    input_imgs = selected_imgs * (1 - selected_masks_tensor)
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
            del selected_imgs, selected_masks_tensor, input_imgs, pred_img, imgs, masks_tensor, frames_PIL, frames, binary_masks, selected_masks
            torch.cuda.empty_cache()

        # Evaluation loop
        ssim, psnr, s_psnr, lpips_score = 0., 0., 0., 0.
        comp_PIL = []
        gt_PIL = []

        # Initialize LPIPS model
        init_lpips_model()

        for f in range(video_length):
            comp = comp_frames[f]
            comp = cv2.cvtColor(np.array(comp), cv2.COLOR_BGR2RGB)

            cv2.imwrite("tmpp.png", comp)
            new_comp = cv2.imread("tmpp.png")
            new_comp = Image.fromarray(cv2.cvtColor(new_comp, cv2.COLOR_BGR2RGB))
            comp_PIL.append(new_comp)
            all_comp_PIL.append(new_comp)  # Add to VFID collection

            # Store ground truth for VFID
            gt = gt_frames[f]
            gt_pil = Image.fromarray(cv2.cvtColor(gt, cv2.COLOR_BGR2RGB))
            gt_PIL.append(gt_pil)
            all_gt_PIL.append(gt_pil)  # Add to VFID collection

            if args.dump_results:
                cv2.imwrite(os.path.join(this_dump_results_dir, "{:04}.png".format(f)), comp)

            # Compute metrics
            gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            ssim += ssim_func(comp, gt_rgb, data_range=255, channel_axis=-1, win_size=65)
            s_psnr += psnr_func(gt_rgb, comp, data_range=255)

            # Compute LPIPS (perceptual similarity)
            # Convert images to tensors normalized to [-1, 1]
            comp_tensor = transforms.ToTensor()(comp).unsqueeze(0) * 2 - 1
            gt_tensor = transforms.ToTensor()(gt_rgb).unsqueeze(0) * 2 - 1
            comp_tensor = comp_tensor.to(device)
            gt_tensor = gt_tensor.to(device)
            
            # Compute perceptual similarity
            lpips_val = compute_lpips(comp_tensor, gt_tensor)
            lpips_score += lpips_val

            # Free up memory
            del comp_tensor, gt_tensor
            torch.cuda.empty_cache()

        ssim_all += ssim
        s_psnr_all += s_psnr
        lpips_all += lpips_score
        video_length_all += video_length
        # if video_no % 5 == 0:
        # print("ssim {}, psnr {}".format(ssim_all / video_length_all, s_psnr_all / video_length_all))

        # Print metrics for this video
        print("Video {}: SSIM {:.4f}, PSNR {:.4f}, LPIPS {:.4f}".format(
            frame_list[video_no].split("/")[-1], 
            ssim / video_length, 
            s_psnr / video_length,
            lpips_score / video_length
        ))
        
        # Print running average
        print("Running average: SSIM {:.4f}, PSNR {:.4f}, LPIPS {:.4f}".format(
            ssim_all / video_length_all, 
            s_psnr_all / video_length_all,
            lpips_all / video_length_all
        ))

        # Compute VFID for this video
        if len(all_comp_PIL) > 0 and len(all_gt_PIL) > 0:
            # Convert to tensors in batches to avoid memory issues
            batch_size = 32  # Process frames in batches for VFID computation
            num_batches = (len(all_comp_PIL) + batch_size - 1) // batch_size
            
            video_output_activations = []
            video_real_activations = []
            
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(all_comp_PIL))
                
                # Process completed frames
                comp_batch = _to_tensors(all_comp_PIL[start:end]).unsqueeze(0).to(device)
                output_activations = get_i3d_activations(comp_batch).cpu().numpy()
                video_output_activations.append(output_activations)
                
                # Process ground truth frames
                gt_batch = _to_tensors(all_gt_PIL[start:end]).unsqueeze(0).to(device)
                real_activations = get_i3d_activations(gt_batch).cpu().numpy()
                video_real_activations.append(real_activations)
                
                del comp_batch, gt_batch
                torch.cuda.empty_cache()
            
            # Concatenate all activations for this video
            if video_output_activations:
                video_output_activations = np.concatenate(video_output_activations, axis=0)
                video_real_activations = np.concatenate(video_real_activations, axis=0)
                
                # Compute mean activation for this video (to avoid memory issues with many videos)
                output_i3d_activations.append(np.mean(video_output_activations, axis=0))
                real_i3d_activations.append(np.mean(video_real_activations, axis=0))

        # Clear CPU memory
        del comp_frames, gt_frames, frame_from_videos, all_comp_PIL, all_gt_PIL, comp_PIL, gt_PIL
        torch.cuda.empty_cache()

    # Compute VFID across all videos
    if len(output_i3d_activations) > 0 and len(real_i3d_activations) > 0:
        output_i3d_activations = np.array(output_i3d_activations)
        real_i3d_activations = np.array(real_i3d_activations)
        fid_score = get_fid_score(real_i3d_activations, output_i3d_activations)
        
        # Final metrics report
        print("\n===== FINAL EVALUATION RESULTS =====")
        print("SSIM: {:.4f}".format(ssim_all / video_length_all))
        print("PSNR: {:.4f}".format(s_psnr_all / video_length_all))
        print("LPIPS: {:.4f} (lower is better)".format(lpips_all / video_length_all))
        print("VFID: {:.4f} (lower is better)".format(fid_score))
        print("====================================")
    else:
        # Final metrics report without VFID
        print("\n===== FINAL EVALUATION RESULTS =====")
        print("SSIM: {:.4f}".format(ssim_all / video_length_all))
        print("PSNR: {:.4f}".format(s_psnr_all / video_length_all))
        print("LPIPS: {:.4f} (lower is better)".format(lpips_all / video_length_all))
        print("VFID: N/A (no valid frames processed)")
        print("====================================")

if __name__ == '__main__':
    main_worker()
