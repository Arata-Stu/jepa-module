import math
from typing import Optional

import torch
torch.set_num_threads(1) # intraop parallelism (this can be a good option)
torch.set_num_interop_threads(1) # interop parallelism


def norm_voxel_grid(voxel_grid: torch.Tensor):
    mask = torch.nonzero(voxel_grid, as_tuple=True)
    if mask[0].size()[0] > 0:
        mean = voxel_grid[mask].mean()
        std = voxel_grid[mask].std()
        if std > 0:
            voxel_grid[mask] = (voxel_grid[mask] - mean) / std
        else:
            voxel_grid[mask] = voxel_grid[mask] - mean
    return voxel_grid


class EventRepresentation:
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor, t_from: Optional[int]=None, t_to: Optional[int]=None):
        raise NotImplementedError


class VoxelGrid(EventRepresentation):
    def __init__(
        self,
        time_bins: Optional[int]=None,
        height: Optional[int]=None,
        width: Optional[int]=None,
        *,
        channels: Optional[int]=None,
    ):
        if time_bins is None:
            time_bins = channels
        elif channels is not None and int(time_bins) != int(channels):
            raise ValueError("time_bins and channels must match when both are provided")

        if time_bins is None:
            raise ValueError("time_bins (or channels) must be provided")
        if height is None or width is None:
            raise ValueError("height and width must be provided")

        assert int(time_bins) >= 1
        assert int(height) > 1
        assert int(width) > 1
        self.time_bins = int(time_bins)
        self.nb_channels = self.time_bins
        self.height = int(height)
        self.width = int(width)

    def get_extended_time_window(self, t0_center: int, t1_center: int):
        dt = self._get_dt(t0_center, t1_center)
        t_start = math.floor(t0_center - dt)
        t_end = math.ceil(t1_center + dt)
        return t_start, t_end

    def _construct_empty_voxel_grid(self):
        return torch.zeros(
            (self.time_bins, self.height, self.width),
            dtype=torch.float,
            requires_grad=False,
            device=torch.device('cpu'))

    def _get_dt(self, t0_center: int, t1_center: int):
        if self.time_bins == 1:
            return 0.0
        assert t1_center > t0_center
        return (t1_center - t0_center)/(self.time_bins - 1)

    def _normalize_time(self, time: torch.Tensor, t0_center: int, t1_center: int):
        if self.time_bins == 1:
            return torch.zeros_like(time, dtype=torch.float32)
        # time_norm < t0_center will be negative
        # time_norm == t0_center is 0
        # time_norm > t0_center is positive
        # time_norm == t1_center is (time_bins - 1)
        # time_norm > t1_center is greater than (time_bins - 1)
        return (time - t0_center)/(t1_center - t0_center)*(self.time_bins - 1)

    @staticmethod
    def _is_int_tensor(tensor: torch.Tensor) -> bool:
        return not torch.is_floating_point(tensor) and not torch.is_complex(tensor)

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor, t0_center: Optional[int]=None, t1_center: Optional[int]=None):
        assert x.device == y.device == pol.device == time.device == torch.device('cpu')
        assert type(t0_center) == type(t1_center)
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1
        assert self._is_int_tensor(time)

        is_int_xy = self._is_int_tensor(x)
        if is_int_xy:
            assert self._is_int_tensor(y)

        voxel_grid = self._construct_empty_voxel_grid()
        ch, ht, wd = self.time_bins, self.height, self.width
        with torch.no_grad():
            value = 2*pol.float()-1
            flat_size = voxel_grid.numel()

            def _safe_put(index_tensor: torch.Tensor, weight_tensor: torch.Tensor, mask_tensor: torch.Tensor):
                index_sel = index_tensor[mask_tensor].long()
                if index_sel.numel() == 0:
                    return
                weight_sel = weight_tensor[mask_tensor]
                valid = (index_sel >= 0) & (index_sel < flat_size)
                if bool(torch.any(valid)):
                    voxel_grid.put_(index_sel[valid], weight_sel[valid], accumulate=True)

            if ch == 1:
                if is_int_xy:
                    mask = (x >= 0) & (x < wd) & (y >= 0) & (y < ht)
                    index = wd * y.long() + x.long()
                    index_sel = index[mask]
                    if index_sel.numel() > 0:
                        valid = (index_sel >= 0) & (index_sel < (ht * wd))
                        if bool(torch.any(valid)):
                            voxel_grid[0].put_(index_sel[valid], value[mask][valid], accumulate=True)
                else:
                    x0 = x.floor().int()
                    y0 = y.floor().int()
                    for xlim in [x0, x0+1]:
                        for ylim in [y0, y0+1]:
                            mask = (xlim < wd) & (xlim >= 0) & (ylim < ht) & (ylim >= 0)
                            interp_weights = value * (1 - (xlim - x).abs()) * (1 - (ylim - y).abs())
                            index = wd * ylim.long() + xlim.long()
                            index_sel = index[mask]
                            if index_sel.numel() > 0:
                                valid = (index_sel >= 0) & (index_sel < (ht * wd))
                                if bool(torch.any(valid)):
                                    voxel_grid[0].put_(
                                        index_sel[valid],
                                        interp_weights[mask][valid],
                                        accumulate=True,
                                    )
                return voxel_grid

            t0_center = t0_center if t0_center is not None else time[0]
            t1_center = t1_center if t1_center is not None else time[-1]
            if int(t1_center) <= int(t0_center):
                t1_center = int(t0_center) + 1
            t_norm = self._normalize_time(time, t0_center, t1_center)

            t0 = t_norm.floor().int()

            if is_int_xy:
                for tlim in [t0,t0+1]:
                    mask = (x >= 0) & (x < wd) & (y >= 0) & (y < ht) & (tlim >= 0) & (tlim < ch)
                    interp_weights = value * (1 - (tlim - t_norm).abs())

                    index = ht * wd * tlim.long() + \
                            wd * y.long() + \
                            x.long()

                    _safe_put(index, interp_weights, mask)
            else:
                x0 = x.floor().int()
                y0 = y.floor().int()
                for xlim in [x0,x0+1]:
                    for ylim in [y0,y0+1]:
                        for tlim in [t0,t0+1]:

                            mask = (xlim < wd) & (xlim >= 0) & (ylim < ht) & (ylim >= 0) & (tlim >= 0) & (tlim < ch)
                            interp_weights = value * (1 - (xlim-x).abs()) * (1 - (ylim-y).abs()) * (1 - (tlim - t_norm).abs())

                            index = ht * wd * tlim.long() + \
                                    wd * ylim.long() + \
                                    xlim.long()

                            _safe_put(index, interp_weights, mask)

        return voxel_grid
