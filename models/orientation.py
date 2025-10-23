import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.orientation_module import OrientNet


class DataAugment(nn.Module):
	"""Subset of SE-ORNet data augmentation routines for rotation/noise."""

	def __init__(
		self,
		operations=None,
		flip_probability: float = 0.0,
		scale_range=None,
		rotate_nbins: int = 8,
		noise_variance: float = 0.0001,
	):
		super().__init__()
		if operations is None:
			operations = ["noise"]
		if scale_range is None:
			scale_range = [0.95, 1.05]
		self.operations = operations
		self.flip_probability = flip_probability
		self.scale_range = scale_range
		self.rotate_nbins = rotate_nbins
		self.noise_variance = noise_variance

	def forward(self, batch_data: torch.Tensor):
		device = batch_data.device
		out = batch_data.clone()
		rotated_gt: Optional[torch.Tensor] = None

		if "scale" in self.operations:
			scales = torch.empty(
				batch_data.size(0), device=device, dtype=batch_data.dtype
			).uniform_(self.scale_range[0], self.scale_range[1])
			out = out * scales.view(-1, 1, 1)
		if "rotate" in self.operations:
			out, rotated_gt = rotate_by_z_axis(out, self.rotate_nbins)
		if "noise" in self.operations:
			noise = torch.randn_like(out) * math.sqrt(self.noise_variance)
			out = out + noise
		if rotated_gt is None:
			return out
		return out, rotated_gt


def rotate_by_z_axis(batch_data: torch.Tensor, nbins: int = 8):
	device = batch_data.device
	angle_bins = (
		torch.arange(nbins, device=device, dtype=batch_data.dtype) * 2 * math.pi / nbins
		- math.pi / 4
	)
	rotated_gt = torch.randint(nbins, (batch_data.size(0),), device=device)
	rotation_angle = angle_bins[rotated_gt]
	rotated = torch.zeros_like(batch_data)
	for k in range(batch_data.size(0)):
		cosval = torch.cos(rotation_angle[k])
		sinval = torch.sin(rotation_angle[k])
		rot = torch.tensor(
			[[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]], device=device
		)
		centroid = batch_data[k].mean(dim=0, keepdim=True)
		centered = batch_data[k] - centroid
		rotated[k] = torch.mm(centered, rot) + centroid
	return rotated, rotated_gt


class FocalLoss(nn.Module):
	def __init__(self, class_num: int, gamma: float = 2.0):
		super().__init__()
		self.class_num = class_num
		self.gamma = gamma

	def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
		probs = F.softmax(inputs, dim=-1)
		targets = targets.view(-1, 1)
		one_hot = torch.zeros_like(probs).scatter_(1, targets, 1.0)
		pt = (probs * one_hot).sum(dim=1)
		log_pt = torch.log(pt + 1e-8)
		loss = -(1 - pt) ** self.gamma * log_pt
		return loss.mean()


class OrientationAligner(nn.Module):
	"""Wraps OrientNet to align point clouds and compute orientation losses."""

	def __init__(
		self,
		orient_dims=None,
		angle_bins: int = 8,
		angle_weight: float = 0.4,
		domain_weight: float = 1.0,
		device: Optional[torch.device] = None,
	):
		super().__init__()
		if orient_dims is None:
			orient_dims = [3, 64, 128, 256]
		self.angle_bins = angle_bins
		self.angle_weight = angle_weight
		self.domain_weight = domain_weight
		self.device = device

		self.orientnet = OrientNet(
			input_dims=orient_dims,
			output_dim=256,
			latent_dim=256,
			mlps=[256, 128, 128],
			num_class=angle_bins,
		)

		self.src_aug = DataAugment(operations=["noise"], noise_variance=0.0001)
		self.tgt_aug = DataAugment(
			operations=["rotate", "noise"],
			rotate_nbins=angle_bins,
			noise_variance=0.0001,
		)
		self.domain_loss = FocalLoss(class_num=2, gamma=3)

		angle_values = torch.arange(angle_bins, dtype=torch.float32) * 2 * math.pi / angle_bins
		angle_values = angle_values - math.pi / 4
		self.register_buffer("angle_lookup", angle_values)

	def rotate_point_cloud_by_angle(self, xyz: torch.Tensor, angles: torch.Tensor):
		rotated = torch.zeros_like(xyz)
		for k in range(xyz.size(0)):
			rotation_angle = -self.angle_lookup[angles[k]]
			cosval = torch.cos(rotation_angle)
			sinval = torch.sin(rotation_angle)
			rot = torch.tensor(
				[[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]],
				device=xyz.device,
				dtype=xyz.dtype,
			)
			centroid = xyz[k].mean(dim=0, keepdim=True)
			centered = xyz[k] - centroid
			rotated[k] = torch.mm(centered, rot) + centroid
		return rotated

	def forward(
		self,
		source: torch.Tensor,
		target: torch.Tensor,
		mode: str = "train",
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		source_aug = source
		target_aug = target
		rotated_gt = None
		if mode in {"train", "val"}:
			source_aug = self.src_aug(source)
			target_aug, rotated_gt = self.tgt_aug(target)

		outputs_clean = self.orientnet(source, target)
		outputs_aug = self.orientnet(source_aug, target_aug)

		angle_index = outputs_clean["angle_x"].argmax(dim=-1)
		source_aligned = self.rotate_point_cloud_by_angle(source, angle_index)

		angle_loss = torch.zeros(1, device=source.device)
		domain_loss = torch.zeros(1, device=source.device)

		if rotated_gt is not None:
			angle_loss = F.cross_entropy(outputs_aug["angle_x"], rotated_gt)
			domain_real = outputs_clean["global_d_pred"]
			domain_aug = outputs_aug["global_d_pred"]
			labels_real = torch.zeros(domain_real.size(0), device=source.device, dtype=torch.long)
			labels_aug = torch.ones(domain_aug.size(0), device=source.device, dtype=torch.long)
			domain_loss = self.domain_loss(domain_real, labels_real) + self.domain_loss(domain_aug, labels_aug)

		total_angle = angle_loss * self.angle_weight
		total_domain = domain_loss * self.domain_weight
		return source_aligned, target, total_angle, total_domain


__all__ = ["OrientationAligner"]
