#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.
import time
import random
import logging
import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import MinkowskiEngine as ME
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.nn import functional as F

from models.reloc import Regressor
from models.loss import REG_Criterion, CriterionPose, barlow_twins_loss
from datasets.lidarloc import LiDARLocDataset
from datasets.base_loader import CollationFunctionFactory
from utils.pose_util import find_anchor_positive_pairs_indices

from pathlib import Path
from distutils.util import strtobool
from tqdm import tqdm

_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))

def set_seed(seed):
    """
    Seed all sources of randomness.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)

# Used to seed workers in a reproducible manner.
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)


class TripletGroupedBatchSampler(sampler.Sampler):
    """
    Yields a list of scan indices with grouping order:
      [a0, a0_closest1, a0_closest2, a0_closest3, a1, a1_closest1, ...]
    so the dataset can keep its size and just follow the provided order.
    """

    def __init__(self, dataset, batch_size_scans: int, group_size: int, generator: torch.Generator):
        self.dataset = dataset
        self.batch_size_scans = int(batch_size_scans)
        self.group_size = int(group_size)
        self.generator = generator
        self.anchors_per_batch = self.batch_size_scans // self.group_size

        # Keep optimizer step count per epoch comparable to non-grouped setting.
        self.num_batches = max(
            1, (len(self.dataset) + self.batch_size_scans - 1) // self.batch_size_scans
        )

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        n = len(self.dataset)
        for _ in range(self.num_batches):
            # Sample anchors with replacement for simplicity (RandomSampler-permutation would be longer).
            anchors = torch.randint(0, n, (self.anchors_per_batch,), generator=self.generator).tolist()
            batch = []
            for a in anchors:
                # closest = list(self.dataset.closest_scan_global_indices.get(a, []))
                closest = sorted(list(self.dataset.closest_scan_global_indices.get(a, [])))
                k = max(0, self.group_size - 1)

                # Pick memory scans from *other sequences* randomly.
                # - If closest has >= k entries: sample without replacement.
                # - If closest has < k entries: sample with replacement.
                if k <= 0:
                    mem = []
                elif len(closest) == 0:
                    mem = [a] * k  # fallback: reuse anchor
                elif len(closest) >= k:
                    sel = torch.randperm(len(closest), generator=self.generator)[:k].tolist()
                    mem = [closest[i] for i in sel]
                else:
                    sel = torch.randint(0, len(closest), (k,), generator=self.generator).tolist()
                    mem = [closest[i] for i in sel]

                group = [a] + mem
                while len(group) < self.group_size:
                    group.append(a)
                batch.extend(group)
            yield batch

class Trainer:
    def __init__(self, options):
        self.options = options

        self.device = torch.device('cuda')

        # DataLoader multiprocessing can exhaust file descriptors over long runs on some systems.
        # This sharing strategy reduces pressure on FD/semaphore resources.
        try:
            mp.set_sharing_strategy('file_system')
        except Exception:
            pass

        # Setup randomness for reproducibility.
        self.base_seed = 2089
        set_seed(self.base_seed)

        # Used to generate batch indices.
        self.batch_generator = torch.Generator()
        self.batch_generator.manual_seed(self.base_seed + 1023)

        # Dataloader generator, used to seed individual workers by the dataloader.
        self.loader_generator = torch.Generator()
        self.loader_generator.manual_seed(self.base_seed + 511)

        # Generator used to sample random features (runs on the GPU).
        self.sampling_generator = torch.Generator(device=self.device)
        self.sampling_generator.manual_seed(self.base_seed + 4095)

        # Generator used to permute the feature indices during each training epoch.
        self.training_generator = torch.Generator()
        self.training_generator.manual_seed(self.base_seed + 8191)

        # Generator for global feature noise
        self.gn_generator = torch.Generator(device=self.device)
        self.gn_generator.manual_seed(self.base_seed + 24601)

        self.iteration = 0
        self.training_start = None
        self.num_data_loader_workers = 12

        # Create dataset.
        self.dataset = LiDARLocDataset(
            root_dir=self.options.scene,  
            train=True,  
            sample_cls=self.options.sample_cls, 
            augment=self.options.use_aug,  # use augmentation?
            aug_rotation=self.options.aug_rotation,  # max rotation
            aug_translation=self.options.aug_translation,  # max translation
            generate_clusters=self.options.generate_clusters,   # generate new classification label
            level_clusters=self.options.level_cluster,
            voxel_size=self.options.voxel_size,  # voxel size of point cloud
            group_size=(self.options.triplet_group_size if self.options.sample_cls else 1),
        )

        # Create network using the state dict of the pretrained encoder.
        encoder_state_dict = torch.load(self.options.encoder_path, map_location="cpu")
        _logger.info(f"Loaded pretrained encoder from: {self.options.encoder_path}")

        # REG
        if not self.options.sample_cls:
            # # Setup loss function.
            self.loss = REG_Criterion()
            sampler = None
            shuffle = True
                
            # Load classification weights
            classifier_state_dict = torch.load(self.options.classifier_path, map_location="cpu")
            collation_fn = CollationFunctionFactory(collation_type='collate_pair_reg')
            self.dataloader = DataLoader(dataset=self.dataset,
                                         batch_size=self.options.batch_size,
                                         pin_memory=True,
                                         shuffle=shuffle,
                                         num_workers=self.num_data_loader_workers,
                                         persistent_workers=self.num_data_loader_workers > 0,
                                         collate_fn=collation_fn,
                                         sampler=sampler,
                                         )
        # CLS
        else:
            classifier_state_dict = None
            self.pose_loss = CriterionPose(rot_weight=100)
            self.batch_sampler = TripletGroupedBatchSampler(
                self.dataset,
                batch_size_scans=self.options.batch_size,
                group_size=self.options.triplet_group_size,
                generator=self.batch_generator,
            )
            
            collation_fn = CollationFunctionFactory(collation_type='collate_pair_cls')
            self.dataloader = DataLoader(dataset=self.dataset,
                                         batch_sampler=self.batch_sampler,
                                         worker_init_fn=seed_worker,
                                         generator=self.loader_generator,
                                         pin_memory=True,
                                         num_workers=self.num_data_loader_workers,
                                         persistent_workers=True,
                                         collate_fn=collation_fn,
                                         )
            
        self.regressor = Regressor.create_from_encoder(
            encoder_state_dict=encoder_state_dict,
            classifier_state_dict=classifier_state_dict,
            num_head_blocks=self.options.num_head_blocks,
            mlp_ratio=self.options.mlp_ratio,
            level_clusters=self.options.level_cluster,
            sample_cls=self.options.sample_cls,
        )
        self.regressor = self.regressor.to(self.device)

        # Freeze encoder parameters
        for param in self.regressor.encoder.parameters():
            param.requires_grad = False

        if not self.options.sample_cls: # REG
            # Freeze classifier head parameters
            for param in self.regressor.cls_heads.parameters():
                param.requires_grad = False
            steps_per_epoch = len(self.dataset) // self.options.batch_size
        else:
            steps_per_epoch = max(1, (len(self.dataset) + self.options.batch_size - 1) // self.options.batch_size)

        train_parameter = filter(lambda p: p.requires_grad, self.regressor.parameters())
        
        # REG
        if not self.options.sample_cls:
            # Setup optimization parameters.
            self.optimizer = optim.AdamW(train_parameter, lr=self.options.learning_rate_min)
            # Setup learning rate scheduler.
            self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                        max_lr=self.options.learning_rate_max,
                                                        epochs=self.options.epochs,
                                                        steps_per_epoch=steps_per_epoch,
                                                        cycle_momentum=False)
        else: # Training the global embedding module
            self.optimizer_cls = optim.AdamW(train_parameter, lr=self.options.learning_rate_min)
            self.scheduler_cls = optim.lr_scheduler.OneCycleLR(self.optimizer_cls,
                                                        max_lr=self.options.learning_rate_max,
                                                        epochs=self.options.epochs,
                                                        steps_per_epoch=steps_per_epoch,
                                                        cycle_momentum=False)

        # Gradient scaler in case we train with half precision.
        self.scaler = GradScaler(enabled=self.options.use_half)

        # Compute total number of iterations.
        self.iterations = self.options.epochs * steps_per_epoch
        self.iterations_output = 100  # print loss every n iterations, and (optionally) write a visualisation frame

        # Will be filled at the beginning of the training process.
        self.training_buffer = None
        
    def train(self):
        """
        Main training method.
        """

        creating_buffer_time = 0.
        training_time = 0.

        if self.options.sample_cls:
            # For CLS training
            self.training_start = time.time()

            # Train the regression head.
            for self.epoch in range(self.options.epochs):
                epoch_start_time = time.time()
                self.cls_run_epoch()
                training_time += time.time() - epoch_start_time
            end_time = time.time()

            # Save trained model.
            self.save_model(self.epoch, 'cls')
            _logger.info(f'Done without errors. '
                         f'Creating buffer time: {creating_buffer_time / 60:.1f} minutes. '
                         f'Training time: {training_time / 60 :.1f} minutes. '
                         f'Total time: {(end_time - self.training_start) / 60 :.1f} minutes.')

        else:
            # For REG training
            self.training_start = time.time()
            for self.epoch in range(self.options.epochs):
                self.reg_run_epoch()
            end_time = time.time()
            # Save trained model.
            self.save_model(self.epoch, 'reg')
            _logger.info(f'Done without errors. '
                         f'Training time: {(end_time - self.training_start) / 60:.1f} minutes. ')
            

    def cls_run_epoch(self):
        """
        Train the global embedding module
        MLP-Mixer based Global Feature Extraction + Contrastive Learning
        """
        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False
        _logger.info("Starting training MLP-Mixer.")

        # The encoder is pretrained, so we don't compute any gradient.
        tqdm_loader = tqdm(self.dataloader, total=len(self.dataloader))

        for step, batch in enumerate(tqdm_loader):

            # Copy to device.
            features = batch['sinput_F'].to(self.device, non_blocking=True)
            coordinates = batch['sinput_C'].to(self.device, non_blocking=True)
            pcs_tensor = ME.SparseTensor(features[:, :3], coordinates)
            poses = batch['pose'].to(self.device, non_blocking=True) # [B, 1,  6] -> [tx, ty, tz, qx, qy, qz] 
            lbl = batch['lbl'].to(self.device, non_blocking=True).squeeze().squeeze()
            batch_size = lbl.size(0)
            # Generate group_ids based on batch order (since Sampler guarantees [a, p1, p2, p3] grouping)
            group_ids = torch.arange(batch_size, device=self.device) // self.options.triplet_group_size
            gt_t = poses[:, 0, :3]
            gt_q = poses[:, 0, 3:]
            
            # Compute Local Features
            self.regressor.eval()
            with torch.no_grad():
                features = self.regressor.get_features(pcs_tensor)
                if self.options.use_half:
                    featuresF = features.F.half()
                else:
                    featuresF = features.F
                # batch indices
                batch_nums = features.C[:, 0]
            
            anchor_indices, positive_indices = find_anchor_positive_pairs_indices(
                group_ids, device=self.device
            )

            self.regressor.train()
            with autocast(enabled=self.options.use_half):
                pred_t, pred_q, projected_embs = self.regressor.get_poses_and_projection_embedding(featuresF, batch_nums) # [B, 3], [B, 3], [B, 256]
            
            pose_loss = self.pose_loss(pred_t, pred_q, gt_t, gt_q)
            projected_emb_query = projected_embs[anchor_indices]
            projected_emb_pos = projected_embs[positive_indices]
            barlow_loss = barlow_twins_loss(projected_emb_query, projected_emb_pos)
            loss = pose_loss + 0.0001 * barlow_loss 

            # Optimization steps.
            optimizer_step = self.optimizer_cls._step_count
            self.optimizer_cls.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer_cls)
            self.scaler.update()
            torch.cuda.empty_cache()

            if self.iteration % self.iterations_output == 0:
                time_since_start = time.time() - self.training_start
                _logger.info(f'Iteration: {self.iteration:6d} / Epoch {self.epoch:03d}|{self.options.epochs:03d}, '
                            f'Loss: {loss:.4f}, Time: {time_since_start:.2f}s')

            if optimizer_step < self.optimizer_cls._step_count:
                self.scheduler_cls.step()

            self.iteration += 1


    def reg_run_epoch(self):
        '''
        Train the local feature enhancement module and scene coordinate regression head
        '''
        reg_pool = ME.MinkowskiAvgPooling(kernel_size=8, stride=8, dimension=3)
        tqdm_loader = tqdm(self.dataloader, total=len(self.dataloader))

        # The number point of sampling
        num_point = 256

        for step, batch in enumerate(tqdm_loader):
            
            features = batch['sinput_F'].to(self.device, non_blocking=True)
            coordinates = batch['sinput_C'].to(self.device, non_blocking=True)
            idx = batch['idx'].to(self.device)
            label = batch['lbl']
            batch_size = label.size(0)

            pcs_tensor = ME.SparseTensor(features[:, :3], coordinates)
            pcs_tensor_s8 = ME.SparseTensor(features, coordinates)
            
            self.regressor.eval()
            with torch.no_grad():
                # Extract local features
                out_feat = self.regressor.get_features(pcs_tensor)
                # half precision
                if self.options.use_half:
                    featuresF = out_feat.F.half()
                else:
                    featuresF = out_feat.F
                # batch indices
                batch_nums = out_feat.C[:, 0]

                with autocast(enabled=self.options.use_half):
                    global_descs = self.regressor.get_aggregated_features(featuresF, batch_nums)
                    
                # Normalize global descriptors
                global_descs = F.normalize(global_descs, p=2, dim=-1)
                # Ground Truth for scene coordinate regression
                ground_truth = reg_pool(pcs_tensor_s8)
                # [:3] point cloud in the LiDAR coordinate; [3:6] corresponding point cloud in the world coordinate
                gt_sup_point = ground_truth.features_at_coordinates(out_feat.C.float())[:, 3:6]

            # Random Sampling
            random_indices = torch.randperm(len(batch_nums), generator=self.training_generator)[:(batch_size * num_point)]

            self.regressor.train()
            with torch.set_grad_enabled(True):
                # Apply self-attention
                attn_feat = self.regressor.reg_heads.apply_self_attention(featuresF.float(), batch_nums)
                # Sample features
                global_feats_sampled = global_descs[batch_nums.long()][random_indices]
                local_feats_sampled = attn_feat[random_indices]                
                # Add Gaussian Noise to global features same as LightLoc
                noise = torch.empty_like(global_feats_sampled).normal_(mean=0, std=0.1, generator=self.gn_generator)
                global_feats_sampled = global_feats_sampled + noise
                features_sampled = torch.cat([global_feats_sampled, local_feats_sampled], dim=1)
                target_part = gt_sup_point[random_indices]

                with autocast(enabled=self.options.use_half):
                    pred_scene = self.regressor.get_scene_coordinates(features_sampled)

            loss = self.loss(pred_scene[:(batch_size * num_point)], target_part[:(batch_size * num_point)])
            optimizer_step = self.optimizer._step_count

            # Optimization steps.
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            # Clip the gradient.
            torch.nn.utils.clip_grad_norm_(self.regressor.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            torch.cuda.empty_cache()

            if optimizer_step < self.optimizer._step_count < self.scheduler.total_steps:
                self.scheduler.step()

            if step % self.iterations_output == 0:
                time_since_start = time.time() - self.training_start
                _logger.info(f'Epoch {self.epoch:03d}|{self.options.epochs:03d}, '
                             f'Loss: {loss:.4f}, Time: {time_since_start:.2f}s')
        
    def save_model(self, epoch, cor):
        if cor == 'cls':
            state_dict = self.regressor.cls_heads.state_dict()
        else:
            state_dict = self.regressor.reg_heads.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = state_dict[k]

        if self.dataset.scene == 'QEOxford':
            torch.save(state_dict, 'log/' + str(epoch) + '_' + cor + '_QEOxford.pth')
        elif self.dataset.scene == 'NCLT':
            torch.save(state_dict, 'log/' + str(epoch) + '_' + cor + '_NCLT.pth')
        else:
            raise ValueError(f"Invalid dataset name: {self.dataset_name}")

        _logger.info(f"Saved trained head weights to: {self.options.output_map_file}")


if __name__ == '__main__':

    # CUDA_VISIBLE_DEVICES=1 python train_with_barlow_attention_ver3_50epochs_trial7.py --scene=/data/Oxford/ --sample_cls=True --voxel_size=0.25 --epochs=50 --batch_size=512 --triplet_group_size=4
    # CUDA_VISIBLE_DEVICES=1 python train_with_barlow_attention_ver3_50epochs_trial7.py --scene=/data/Oxford/ --classifier_path=log/49_cls_QEOxford.pth --voxel_size=0.25 --epochs=25 --batch_size=256 --triplet_group_size=1

    # Setup logging levels.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Fast training of a sample classification network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data path
    parser.add_argument('--scene', type=Path, default='/data/Oxford',
                        help='path to a scene in the dataset folder')

    # Output path
    parser.add_argument('--output_map_file', type=Path, default='log/',
                        help='target file for the trained network')

    # Encoder path
    parser.add_argument('--encoder_path', type=Path, default='log/Backbone.pth',
                        help='file containing pre-trained encoder weights')

    # Classifier path
    parser.add_argument('--classifier_path', type=Path, default='log/49_cls_QEOxford.pth',
                        help='file containing trained classifier weights')

    # Train classification or regression
    parser.add_argument('--sample_cls', type=_strtobool, default=False,
                        help='True: train global embedding module; False: train local feature enhancement module and scene coordinate regression head')

    # Dimension of global descriptor
    parser.add_argument('--level_cluster', type=int, default=512,
                        help='the dimension of global descriptor')

    parser.add_argument('--generate_clusters', type=_strtobool, default=False,
                        help='Generate clusters for classification training')

    # Architecture of scene regression head
    parser.add_argument('--num_head_blocks', type=int, default=2,
                        help='depth of the regression head, defines the map size')

    parser.add_argument('--mlp_ratio', type=int, default=2.0,
                        help='mlp ratio for res blocks')

    # Learn rate
    parser.add_argument('--learning_rate_min', type=float, default=0.0005,
                        help='lowest learning rate of 1 cycle scheduler')

    parser.add_argument('--learning_rate_max', type=float, default=0.005,
                        help='highest learning rate of 1 cycle scheduler')

    # Buffer size, only used for training classification
    parser.add_argument('--training_buffer_size', type=int, default=160000,
                        help='number of samples in the training buffer')

    # Triplet / grouping options (for contrastive learning)
    parser.add_argument('--triplet_group_size', type=int, default=4,
                        help='number of scans per position group (anchor + closest scans from other seqs)')

    # Batch size
    parser.add_argument('--batch_size', type=int, default=256,
                        help='number of samples for each parameter update. classification: 512; regression: 256')

    # Train epoch
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of runs. classification: 50; regression: 25.')

    # Using half-precision for training
    parser.add_argument('--use_half', type=_strtobool, default=False,
                        help='train with half precision')

    # Using data augmentation
    parser.add_argument('--use_aug', type=_strtobool, default=True,
                        help='Use any augmentation.')

    # Max rotation angle (degree)
    parser.add_argument('--aug_rotation', type=int, default=10,
                        help='max rotation angle')

    # Max translation distance (meter)
    parser.add_argument('--aug_translation', type=int, default=1,
                        help='max translation meter')

    # Voxel size for sparse conv
    parser.add_argument('--voxel_size', type=float, default=0.25,
                        help='Oxford 0.25 NCLT 0.30')

    options = parser.parse_args()

    # For reproducibility
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    torch.set_num_threads(5)
    trainer = Trainer(options)
    trainer.train()
