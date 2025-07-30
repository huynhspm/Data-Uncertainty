from typing import List
from pathlib import Path
import json
from models.model_configs import instantiate_model
import torch
from training.eval_loop import CFGScaledModel
from flow_matching.solver.ode_solver import ODESolver

import torchvision.datasets as datasets
from datasets import (LIDCDataset, ISIC2016Dataset, 
                    ISIC2018Dataset, MSMRIDataset, MMISDataset,
                    QUBIQPanDataset, QUBIQPanLesDataset)
from torch.utils.data import DataLoader
from training.data_transform import get_train_transform, TransformDataset

from tqdm import tqdm
from functools import partial

import os
import numpy as np
import torch
import argparse
import torch.serialization
import matplotlib.pyplot as plt

def encode(x):
    x = torch.clamp(x * 255, min=0.0, max=255.0)
    x = (x - 127.5) / 127.5
    return x

def decode(x):
    x = torch.clamp(x * 0.5 + 0.5, min=0.0, max=1.0)
    x = torch.floor(x * 255) / 255.0
    return x

def compute_metric(preds: List[torch.Tensor], gts: List[torch.Tensor], batch: int):
    """_summary_
    Args:
        preds (_type_): List[Tensor(b, w, h) x n]
        gts (_type_): List[Tensor(b, w, h) x m]
    """
    def compute_iou(output: torch.Tensor, target: torch.Tensor):
        """_summary_
        
        Args:
            output: Tensor(b, c, w, h) or Tensor(c, w, h) or Tensor(w, h)
            target: Tensor(b, c, w, h) or Tensor(c, w, h) or Tensor(w, h)
        """
        smooth = 1e-5
        assert output.shape == target.shape, "Output and target must have the same shape"
        output = output.data.cpu().numpy()
        target = target.data.cpu().numpy()
        output_ = output > 0.5
        target_ = target > 0.5
        
        intersection = (output_ & target_).sum(axis=(-2, -1))
        union = (output_ | target_).sum(axis=(-2, -1))
        iou = (intersection + smooth) / (union + smooth)
        
        return iou.mean()

    def compute_dice(output: torch.Tensor, target: torch.Tensor):
        """_summary_
        Args:
            output: Tensor(b, c, w, h) or Tensor(c, w, h) or Tensor(w, h)
            target: Tensor(b, c, w, h) or Tensor(c, w, h) or Tensor(w, h)
        """
        smooth = 1e-5
        assert output.shape == target.shape, "Output and target must have the same shape"
        output = output.data.cpu().numpy()
        target = target.data.cpu().numpy()
        output_ = output > 0.5
        target_ = target > 0.5
        
        intersection = (output_ * target_).sum(axis=(-2, -1))
        total = output_.sum(axis=(-2, -1)) + target_.sum(axis=(-2, -1))
        dice = (2. * intersection + smooth) / (total + smooth)
        
        return dice.mean()

    def compute_ged(preds: torch.Tensor, gts: torch.Tensor):
        """_summary_
        Args:
            preds (_type_): Tensor(n, w, h)
            gts (_type_): Tensor(m, w, h)
        """
        n, m = preds.shape[0], gts.shape[0]
        d1, d2, d3 = 0, 0, 0
        
        for i in range(n):
            for j in range(m):
                d1 = d1 + (1 - compute_iou(preds[i], gts[j]))
        
        for i in range(n):
            for j in range(n):
                d2 = d2 + (1 - compute_iou(preds[i], preds[j]))
        
        for i in range(m):
            for j in range(m):
                d3 = d3 + (1 - compute_iou(gts[i], gts[j]))
        
        d1, d2, d3 = (2*d1)/(n*m), d2/(n*n), d3/(m*m)
        ged = d1 - d2 - d3
        return ged

    def compute_max_dice(preds: torch.Tensor, gts: torch.Tensor):
        """_summary_
        Args:
            preds (_type_): Tensor(n, w, h)
            gts (_type_): Tensor(m, w, h)
        """
        max_dice = 0
        for gt in gts:
            dices = [compute_dice(pred, gt) for pred in preds]
            max_dice += max(dices)
        return max_dice / len(gts)

    def compute_sncc(pred_samples, gt_annotations):
        """
        pred_samples: Tensor of shape (N_samples, C, H, W) - predicted masks
        gt_annotations: Tensor of shape (N_annotators, C, H, W) - ground-truth masks
        """
        def cross_entropy_map(pred, target, eps=1e-6):
            """Compute per-pixel CE (no reduction)."""
            pred = pred.clamp(min=eps, max=1 - eps)
            return -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))

        def normalized_cross_correlation(x, y, eps=1e-6):
            """Compute Normalized Cross Correlation between two maps."""
            x_mean = torch.mean(x)
            y_mean = torch.mean(y)
            numerator = torch.sum((x - x_mean) * (y - y_mean))
            denominator = torch.sqrt(torch.sum((x - x_mean)**2) * torch.sum((y - y_mean)**2)) + eps
            return numerator / denominator

        mean_pred = torch.mean(pred_samples, dim=0)  # shape: (C, H, W)
        
        # Compute CE(bar_s, s) for each sample
        ce_bar_s_s = [cross_entropy_map(mean_pred, s) for s in pred_samples]
        ce_bar_s_s = torch.stack(ce_bar_s_s)  # (N_samples, C, H, W)
        mean_ce_bar_s_s = torch.mean(ce_bar_s_s, dim=0)  # (C, H, W)

        sncc_scores = []
        for y in gt_annotations:
            ce_y_s = [cross_entropy_map(y, s) for s in pred_samples]
            ce_y_s = torch.stack(ce_y_s)
            mean_ce_y_s = torch.mean(ce_y_s, dim=0)
            
            # Compute NCC between the two maps
            ncc = normalized_cross_correlation(mean_ce_bar_s_s, mean_ce_y_s)
            sncc_scores.append(ncc)

        return torch.mean(torch.tensor(sncc_scores))

    ged, sncc, max_dice, dice, iou = 0, 0, 0, 0, 0

    preds = torch.stack(preds, dim=1) # b, n, w, h
    gts = torch.stack(gts, dim=1) # b, m, w, h
    # for batch
    for _preds, _gts in zip(preds, gts):
        # _preds: n, w, h
        # _gts: m, w, h
        ged += compute_ged(_preds, _gts)
        max_dice += compute_max_dice(_preds, _gts)
        sncc += compute_sncc(_preds, _gts)

        pred = _preds.mean(dim=0) # w, h
        gt = _gts.mean(dim=0) # w, h
        dice += compute_dice(pred, gt)
        iou += compute_iou(pred, gt)
    
    batch = preds.shape[0]
    return ged/batch, sncc / batch, max_dice/batch, dice/batch, iou/batch

def get_dataset(args_dict, sample_resolution):
    if args_dict['dataset'] == "mnist":
        dataset = datasets.MNIST(root=args_dict['data_path'],
                            train=False,
                            download=True,
                            transform=get_train_transform())
    else:
        if args_dict['dataset'] == "lidc":
            dataset = LIDCDataset(data_dir=args_dict['data_path'],
                                train_val_test_dir='Val',
                                mask_type="multi")
        elif "isic" in args_dict['dataset']:
            if args_dict['dataset'] == "isic2016":
                dataset = ISIC2016Dataset(data_dir=args_dict['data_path'],
                                        train_val_test_dir='Test')
            elif args_dict['dataset'] == "isic2018":
                dataset = ISIC2018Dataset(data_dir=args_dict['data_path'],
                                        train_val_test_dir='Test')
            args_dict['dataset'] = "isic"
        elif args_dict['dataset'] == "msmri":
            dataset = MSMRIDataset(data_dir=args_dict['data_path'],
                                train_val_test_dir='Val',
                                mask_type="multi")
        elif args_dict['dataset'] == "mmis":
            dataset = MMISDataset(data_dir=args_dict['data_path'],
                                train_val_test_dir='Val',
                                mask_type="multi")
        elif args_dict['dataset'] == "qubiq_pan":
            dataset = QUBIQPanDataset(data_dir=args_dict['data_path'],
                                    train_val_test_dir='Val',
                                    mask_type="multi")
        elif args_dict['dataset'] == "qubiq_pan_les":
            dataset = QUBIQPanLesDataset(data_dir=args_dict['data_path'],
                                        train_val_test_dir='Val',
                                        mask_type="multi")
        else:
            raise NotImplementedError(f"Unsupported dataset {args.dataset}")
        
        dataset = TransformDataset(dataset, 
                                height=sample_resolution, 
                                width=sample_resolution, mode='test',)
    return dataset

def save_image(labels, gts, preds, image_folder, batch):
    # labels: Tensor(b, c, w, h)
    # gts: List[Tensor(b, w, h) x n]
    # preds: List[Tensor(b, w, h) x n]
    os.makedirs(image_folder, exist_ok=True)

    for i in range(labels.shape[0]):
        image = labels[i][0].numpy()
        plt.imsave(f"{image_folder}/image_{batch}_{i}.png", image)
        for id in range(len(gts)):
            gt_image = gts[id][i].cpu().numpy()
            plt.imsave(f"{image_folder}/gt_{batch}_{i}_{id}.png", gt_image, cmap="gray")
        for id in range(len(preds)):
            pred_image = preds[id][i].cpu().numpy()
            plt.imsave(f"{image_folder}/pred_{batch}_{i}_{id}.png", pred_image, cmap="gray")

def main(args):
    checkpoint_path = Path(args.checkpoint)

    with open(checkpoint_path.parent / f"{args.filename}.txt", "a", encoding="utf-8") as file:
        file.write(f"checkpoint: {args.checkpoint}\n")
        file.write(f"cfg_scale: {args.cfg_scale}\n")
        file.write(f"n_ensemble: {args.n_ensemble}\n")
        file.write(f"batch_size: {args.batch_size}\n")
        file.write(f"sampling: {args.sampling}\n")
        file.write(f"ode_method: {args.ode_method}\n")
        file.write(f"ode_options: {args.ode_options}\n")
    
    args_filepath = checkpoint_path.parent / 'args.json'
    with open(args_filepath, 'r') as f:
        args_dict = json.load(f)
    # override args
    args_dict["cfg_scale"] = args.cfg_scale
    args_dict["n_ensemble"] = args.n_ensemble
    args_dict["batch_size"] = args.batch_size
    args_dict["ode_method"] = args.ode_method
    args_dict["ode_options"] = args.ode_options

    sample_resolution = args_dict['img_size']
    batch_size = args.batch_size    
    print("batch_size:", batch_size)
    print("sample_resolution:", sample_resolution)
    
    dataset = get_dataset(args_dict, sample_resolution)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model = instantiate_model(
        architechture=args_dict['dataset'] + ("_condition" if args_dict["condition"] else ""), 
        is_discrete='discrete_flow_matching' in args_dict and args_dict['discrete_flow_matching'], 
        use_ema=args_dict['use_ema']) 
    torch.serialization.add_safe_globals([argparse.Namespace]) # for pytorch 2.6
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    device = 'cuda'
    model.to(device=device)
    cfg_weighted_model = CFGScaledModel(model)
    solver = ODESolver(velocity_model=cfg_weighted_model)

    # Create partial functions with common arguments pre-filled
    ode_opts = ode_opts
    sample_partial = partial(
        solver.sample,
        time_grid=torch.tensor([0.0, 1.0], device=device),
        method=args_dict['ode_method'],
        atol=ode_opts['atol'] if 'atol' in ode_opts else 1e-5,
        rtol=ode_opts['rtol'] if 'rtol' in ode_opts else 1e-5,
        step_size=(ode_opts['step_size'] if 'step_size' in ode_opts else None),
        cfg_scale=args_dict['cfg_scale'],
    )

    compute_likelihood_partial = partial(
        solver.compute_likelihood,
        time_grid=torch.tensor([1.0, 0], device=device),
        log_p0=lambda x: torch.zeros_like(x),  # ignore for now
        method=args_dict['ode_method'],
        atol=ode_opts['atol'] if 'atol' in ode_opts else 1e-5,
        rtol=ode_opts['rtol'] if 'rtol' in ode_opts else 1e-5,
        step_size=ode_opts['step_size'] if 'step_size' in ode_opts else None,
        enable_grad=True
    )

    print("Discrete:", args_dict['discrete_flow_matching'])
    print("Condition:", args_dict['condition'])

    dice, ncc, max_dice, iou, ged = {}, {}, {}, {}, {}
    for i in range(args.n_ensemble):
        dice[i], ncc[i], max_dice[i], iou[i], ged[i] = 0, 0, 0, 0, 0

    for batch_id, (samples, images) in enumerate(tqdm(dataloader)):
        labels = images.to(device=device)
        if not isinstance(samples, list):
            samples = [samples]

        gts = [sample.squeeze(dim=1) for sample in samples]
        preds = []

        for i in tqdm(range(args.n_ensemble)):
            x_0 = torch.randn_like(samples[0].to(device=device))
            
            resample = sample_partial(x_init=x_0, label=labels)
            
            preds.append(decode(resample.cpu().squeeze(dim=1)))
            metrics = compute_metric(preds, gts, batch=labels.shape[0])
            ged_iter, sncc_iter, max_dice_iter, dice_iter, iou_iter = metrics
            ged[i] += ged_iter
            sncc[i] += sncc_iter
            max_dice[i] += max_dice_iter
            dice[i] += dice_iter
            iou[i] += iou_iter

            with open(checkpoint_path.parent / f"{args.filename}.txt", "a", encoding="utf-8") as file:
                file.write(f"n_ensemble: {i} --- " 
                            f"ged_iter: {ged_iter} --- "
                            f"sncc_iter: {sncc_iter} --- "
                            f"max_dice_iter: {max_dice_iter} --- "
                            f"dice_iter: {dice_iter} --- "
                            f"iou_iter: {iou_iter}\n")

        save_image(images, gts, preds, checkpoint_path.parent / args.filename, batch_id)


    for i in range(args.n_ensemble):
        dice[i] /= len(dataloader)
        sncc[i] /=  len(dataloader)
        max_dice[i] /= len(dataloader)
        iou[i] /= len(dataloader)
        ged[i] /= len(dataloader)
        with open(checkpoint_path.parent / f"{args.filename}.txt", "a", encoding="utf-8") as file:
            file.write(f"n_ensemble: {i} --- "
                        f"GED: {ged[i]} --- "
                        f"SNCC: {sncc[i]} --- "
                        f"Max_Dice: {max_dice[i]} --- "
                        f"Dice: {dice[i]} --- "
                        f"IoU: {iou[i]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data based on version")
    parser.add_argument("--checkpoint", "-ckpt", default=None, type=str, help="Specify the checkpoint")
    parser.add_argument("--batch_size", default=32, type=int, help="Override Batch size")
    parser.add_argument("--cfg_scale", default=0.3, type=float, help="Override classifier-free guidance scale")
    parser.add_argument("--n_ensemble", default=1, type=int, help="Override number of samples to ensemble")
    parser.add_argument("--filename", default="eval", type=str, help="Specify the eval filename")
    parser.add_argument("--sampling", default="random", type=str, choices=["random", "local"], help="Specify the type of sampling for inference")
    parser.add_argument("--ode_method", default="midpoint", type=str, help="ODE method to use for sampling")
    parser.add_argument("--ode_options", default='{"step_size": 0.01}', type=json.loads, help="ODE solver options. Eg. the midpoint solver requires step-size, dopri5 has no options to set.")
    args = parser.parse_args()
    
    main(args)