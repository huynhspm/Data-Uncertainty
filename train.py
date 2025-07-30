# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.

import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from models.model_configs import instantiate_model
from train_arg_parser import get_args_parser

from training import distributed_mode
from training.data_transform import get_train_transform, TransformDataset
from training.eval_loop import eval_model
from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from training.load_and_save import load_model, save_model
from training.train_loop import train_one_epoch

from datasets.simple_shape import SimpleShapeDataset
from datasets import (LIDCDataset, ISIC2016Dataset, 
                    ISIC2018Dataset, MSMRIDataset, MMISDataset,
                    QUBIQPanDataset, QUBIQPanLesDataset)

logger = logging.getLogger(__name__)


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    distributed_mode.init_distributed_mode(args)

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))
    if distributed_mode.is_main_process():
        args_filepath = Path(args.output_dir) / "args.json"
        logger.info(f"Saving args to {args_filepath}")
        with open(args_filepath, "w") as f:
            json.dump(vars(args), f)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + distributed_mode.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    logger.info(f"Initializing Dataset: {args.dataset}")
    transform_train = get_train_transform()
    if args.dataset == "imagenet":
        dataset_train = datasets.ImageFolder(args.data_path, transform=transform_train)
    elif args.dataset == "cifar10":
        dataset_train = datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform_train,
        )
    elif args.dataset == "simple_shape":
        dataset_train = SimpleShapeDataset(num_samples=args.num_samples, image_size=args.img_size)
    elif args.dataset == "mnist":
        dataset_train = datasets.MNIST(
            root=args.data_path,
            train=True,
            download=True,
            transform=transform_train,
        )
    else:
        if args.dataset == "lidc":
            dataset_train = LIDCDataset(data_dir=args.data_path,
                                        train_val_test_dir='Train',
                                        mask_type=args.mask_type)
            dataset_val = LIDCDataset(data_dir=args.data_path,
                                    train_val_test_dir='Val',
                                    mask_type=args.mask_type)
        elif args.dataset == "msmri":
            dataset_train = MSMRIDataset(data_dir=args.data_path,
                                        train_val_test_dir='Train',
                                        mask_type=args.mask_type)
            dataset_val = MSMRIDataset(data_dir=args.data_path,
                                        train_val_test_dir='Val',
                                        mask_type=args.mask_type)
        elif "isic" in args.dataset:
            if args.dataset == "isic2016":
                dataset_train = ISIC2016Dataset(data_dir=args.data_path,
                                                train_val_test_dir='Train')
                dataset_val = ISIC2016Dataset(data_dir=args.data_path,
                                                train_val_test_dir='Test')
            elif args.dataset == "isic2018":
                dataset_train = ISIC2018Dataset(data_dir=args.data_path,
                                            train_val_test_dir='Train')
                dataset_val = ISIC2016Dataset(data_dir=args.data_path,
                                                train_val_test_dir='Test')
            args.dataset = "isic"
        elif args.dataset == "mmis":
            dataset_train = MMISDataset(data_dir=args.data_path,
                                        train_val_test_dir='Train',
                                        mask_type=args.mask_type)
            dataset_val = MMISDataset(data_dir=args.data_path,
                                    train_val_test_dir='Val',
                                    mask_type=args.mask_type)
        elif args.dataset == "qubiq_pan":
            dataset_train = QUBIQPanDataset(data_dir=args.data_path,
                                        train_val_test_dir='Train',
                                        mask_type=args.mask_type)
            dataset_val = QUBIQPanDataset(data_dir=args.data_path,
                                    train_val_test_dir='Val',
                                    mask_type=args.mask_type)
        elif args.dataset == "qubiq_pan_les":
            dataset_train = QUBIQPanLesDataset(data_dir=args.data_path,
                                        train_val_test_dir='Train',
                                        mask_type=args.mask_type)
            dataset_val = QUBIQPanLesDataset(data_dir=args.data_path,
                                    train_val_test_dir='Val',
                                    mask_type=args.mask_type)
        else:
            raise NotImplementedError(f"Unsupported dataset {args.dataset}")
        dataset_train = TransformDataset(dataset_train, height=args.img_size, width=args.img_size, mode="train")
        dataset_val = TransformDataset(dataset_val, height=args.img_size, width=args.img_size, mode="val")

    logger.info(dataset_train)
    logger.info(dataset_val)
    
    logger.info("Intializing DataLoader")
    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    logger.info(str(sampler_train))

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    logger.info("Initializing Model")
    model = instantiate_model(
        architechture=args.dataset + ("_condition" if args.condition else ""),
        is_discrete=args.discrete_flow_matching,
        use_ema=args.use_ema,
    )

    model.to(device)

    model_without_ddp = model
    logger.info(str(model_without_ddp))

    eff_batch_size = (
        args.batch_size * args.accum_iter * distributed_mode.get_world_size()
    )

    logger.info(f"Learning rate: {args.lr:.2e}")

    logger.info(f"Accumulate grad iterations: {args.accum_iter}")
    logger.info(f"Effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(), lr=args.lr, betas=args.optimizer_betas
    )
    if args.decay_lr:
        lr_schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=args.epochs,
            start_factor=1.0,
            end_factor=1e-8 / args.lr,
        )
    else:
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=args.epochs, factor=1.0
        )

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    loss_scaler = NativeScaler()

    load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        lr_schedule=lr_schedule,
    )

    logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if not args.eval_only:
            train_stats = train_one_epoch(
                model=model,
                data_loader=data_loader_train,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                args=args,
            )
            log_stats = {
                 **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
        else:
            log_stats = {
                "epoch": epoch,
            }

        if args.output_dir and (
            (args.eval_frequency > 0 and (epoch + 1) % args.eval_frequency == 0)
            or args.eval_only
            or args.test_run
        ):
            if not args.eval_only:
                save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )
            if args.distributed:
                data_loader_train.sampler.set_epoch(0)
            if distributed_mode.is_main_process():
                fid_samples = args.fid_samples - (num_tasks - 1) * (
                    args.fid_samples // num_tasks
                )
            else:
                fid_samples = args.fid_samples // num_tasks
            eval_stats = eval_model(
                model,
                data_loader_val,
                device,
                epoch=epoch,
                fid_samples=fid_samples,
                args=args,
            )
            log_stats.update({f"eval_{k}": v for k, v in eval_stats.items()})

        if args.output_dir and distributed_mode.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.test_run or args.eval_only:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
