from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa.models.utils.modules import Block
from jepa.utils.tensors import trunc_normal_


@dataclass
class MAEForwardOutput:
    loss: torch.Tensor
    masked_loss: torch.Tensor
    visible_loss: torch.Tensor
    mask_ratio: float


class VoxelMAE(nn.Module):
    """MAE model for event voxel grids using JEPA ViT as encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        encoder_embed_dim: int,
        img_size: tuple[int, int],
        patch_size: int,
        num_frames: int,
        tubelet_size: int,
        in_chans: int = 1,
        decoder_embed_dim: int = 384,
        decoder_depth: int = 4,
        decoder_num_heads: int = 6,
        mlp_ratio: float = 4.0,
        use_silu: bool = False,
    ) -> None:
        super().__init__()
        if num_frames % tubelet_size != 0:
            raise ValueError(
                f"num_frames must be divisible by tubelet_size, got {num_frames} and {tubelet_size}"
            )
        if img_size[0] % patch_size != 0 or img_size[1] % patch_size != 0:
            raise ValueError(
                f"img_size must be divisible by patch_size, got {img_size} and {patch_size}"
            )
        self.encoder = encoder
        self.encoder_embed_dim = int(encoder_embed_dim)
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.in_chans = int(in_chans)
        self.num_frames = int(num_frames)
        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.grid_t = self.num_frames // self.tubelet_size
        self.grid_h = self.img_size[0] // self.patch_size
        self.grid_w = self.img_size[1] // self.patch_size
        self.num_patches = int(self.grid_t * self.grid_h * self.grid_w)
        self.patch_dim = int(
            self.tubelet_size * self.patch_size * self.patch_size * self.in_chans
        )

        self.decoder_embed = nn.Linear(self.encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim)
        )

        act_layer = nn.SiLU if use_silu else nn.GELU
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    act_layer=act_layer,
                    use_rope=False,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_dim, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert [B, C, T, H, W] voxel tensor to [B, L, patch_dim] patch sequence.
        """
        if x.ndim != 5:
            raise ValueError(f"VoxelMAE expects 5D input [B, C, T, H, W], got {tuple(x.shape)}")
        bsz, chans, timesteps, height, width = x.shape
        if chans != self.in_chans:
            raise ValueError(f"Expected C={self.in_chans}, got C={chans}")
        if timesteps != self.num_frames:
            raise ValueError(f"Expected T={self.num_frames}, got T={timesteps}")
        if height != self.img_size[0] or width != self.img_size[1]:
            raise ValueError(
                f"Expected HxW={self.img_size}, got HxW=({height}, {width})"
            )

        t = timesteps // self.tubelet_size
        h = height // self.patch_size
        w = width // self.patch_size
        x = x.reshape(
            bsz,
            chans,
            t,
            self.tubelet_size,
            h,
            self.patch_size,
            w,
            self.patch_size,
        )
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        x = x.reshape(bsz, t * h * w, self.patch_dim)
        return x

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert [B, L, patch_dim] patch sequence to [B, C, T, H, W] voxel tensor.
        """
        if tokens.ndim != 3:
            raise ValueError(f"Expected [B, L, D], got {tuple(tokens.shape)}")
        bsz, n_tokens, dim = tokens.shape
        if n_tokens != self.num_patches:
            raise ValueError(f"Expected L={self.num_patches}, got L={n_tokens}")
        if dim != self.patch_dim:
            raise ValueError(f"Expected D={self.patch_dim}, got D={dim}")

        x = tokens.reshape(
            bsz,
            self.grid_t,
            self.grid_h,
            self.grid_w,
            self.tubelet_size,
            self.patch_size,
            self.patch_size,
            self.in_chans,
        )
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.reshape(
            bsz,
            self.in_chans,
            self.num_frames,
            self.img_size[0],
            self.img_size[1],
        )
        return x

    def random_masking(
        self,
        batch_size: int,
        mask_ratio: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
        len_keep = max(1, int(self.num_patches * (1.0 - mask_ratio)))

        noise = torch.rand(batch_size, self.num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        mask = torch.ones((batch_size, self.num_patches), device=device)
        mask[:, :len_keep] = 0.0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return ids_keep, ids_restore, mask

    def forward_encoder(
        self,
        x: torch.Tensor,
        ids_keep: torch.Tensor,
    ) -> torch.Tensor:
        # Use JEPA ViT encoder and remove masked tokens through its mask-gather path.
        return self.encoder(x, masks=[ids_keep], training=False)

    def forward_decoder(
        self,
        latent: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        x = self.decoder_embed(latent)
        bsz, keep_tokens, dim = x.shape
        total_tokens = ids_restore.shape[1]
        if total_tokens != self.num_patches:
            raise ValueError(
                f"ids_restore has L={total_tokens}, expected {self.num_patches}"
            )

        mask_tokens = self.mask_token.repeat(bsz, total_tokens - keep_tokens, 1)
        x_all = torch.cat([x, mask_tokens], dim=1)
        gather_index = ids_restore.unsqueeze(-1).repeat(1, 1, dim)
        x_all = torch.gather(x_all, dim=1, index=gather_index)
        x_all = x_all + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x_all, _ = blk(x_all)
        x_all = self.decoder_norm(x_all)
        pred = self.decoder_pred(x_all)
        return pred

    def forward_loss(
        self,
        inputs: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        normalize_targets: bool,
        recon_loss: str,
        loss_on_mask_only: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target = self.patchify(inputs)
        if normalize_targets:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / torch.sqrt(var + 1.0e-6)

        if recon_loss == "smooth_l1":
            per_elem = F.smooth_l1_loss(pred, target, reduction="none")
        elif recon_loss == "mse":
            per_elem = F.mse_loss(pred, target, reduction="none")
        else:
            raise ValueError(f"Unsupported recon_loss={recon_loss}; use mse|smooth_l1")

        per_patch = per_elem.mean(dim=-1)  # [B, L]
        mask_sum = torch.clamp(mask.sum(), min=1.0)
        vis_mask = 1.0 - mask
        vis_sum = torch.clamp(vis_mask.sum(), min=1.0)

        masked_loss = (per_patch * mask).sum() / mask_sum
        visible_loss = (per_patch * vis_mask).sum() / vis_sum
        if loss_on_mask_only:
            loss = masked_loss
        else:
            loss = per_patch.mean()
        return loss, masked_loss, visible_loss

    def forward(
        self,
        inputs: torch.Tensor,
        mask_ratio: float,
        normalize_targets: bool,
        recon_loss: str,
        loss_on_mask_only: bool,
    ) -> MAEForwardOutput:
        bsz = int(inputs.shape[0])
        ids_keep, ids_restore, mask = self.random_masking(
            batch_size=bsz,
            mask_ratio=mask_ratio,
            device=inputs.device,
        )
        latent = self.forward_encoder(inputs, ids_keep=ids_keep)
        pred = self.forward_decoder(latent=latent, ids_restore=ids_restore)
        loss, masked_loss, visible_loss = self.forward_loss(
            inputs=inputs,
            pred=pred,
            mask=mask,
            normalize_targets=normalize_targets,
            recon_loss=recon_loss,
            loss_on_mask_only=loss_on_mask_only,
        )
        return MAEForwardOutput(
            loss=loss,
            masked_loss=masked_loss,
            visible_loss=visible_loss,
            mask_ratio=float(mask.mean().detach().item()),
        )
