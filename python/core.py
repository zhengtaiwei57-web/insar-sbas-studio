"""Small-baseline network inversion, with explicit per-pixel rank checks.

For pair (i,j), y_ij = d_j - d_i. The first epoch is fixed to zero.
Spatial referencing subtracts one common pixel from every interferogram.
Weights are coherence squared (a heuristic, not a calibrated variance).
"""
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


def date_string(value):
    if isinstance(value, bytes):
        value = value.decode()
    value = str(value).replace("-", "")
    return datetime.strptime(value, "%Y%m%d").strftime("%Y%m%d")


def design_matrix(dates, pairs):
    dates = [date_string(d) for d in dates]
    if len(dates) < 3 or dates != sorted(set(dates)):
        raise ValueError("至少需要三个按时间排序、无重复的日期。")
    index = {d: i for i, d in enumerate(dates)}
    a = np.zeros((len(pairs), len(dates)))
    seen = set()
    for k, pair in enumerate(pairs):
        p, q = map(date_string, pair)
        if p not in index or q not in index or p == q:
            raise ValueError(f"无效日期对: {p}, {q}")
        key = tuple(sorted((p, q)))
        if key in seen:
            raise ValueError(f"重复干涉对: {key}")
        seen.add(key)
        a[k, index[p]], a[k, index[q]] = -1, 1
    a = a[:, 1:]
    if np.linalg.matrix_rank(a) != len(dates) - 1:
        raise ValueError("干涉网络不连通，无法反演全部日期。请补充连接干涉对。")
    return a


@dataclass
class Stack:
    dates: list
    pairs: list
    values: np.ndarray
    coherence: np.ndarray
    unit: str = "radian"
    wavelength_m: float = 0.05546576
    phase_sign: int = -1
    mask: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
    components: np.ndarray | None = None

    def validate(self):
        self.dates = [date_string(d) for d in self.dates]
        self.pairs = [tuple(map(date_string, p)) for p in self.pairs]
        a = design_matrix(self.dates, self.pairs)
        self.values = np.asarray(self.values, dtype=np.float64)
        self.coherence = np.asarray(self.coherence, dtype=np.float64)
        if self.values.ndim != 3 or self.values.shape[0] != len(self.pairs):
            raise ValueError("values 必须为 [干涉对, 行, 列]。")
        if self.coherence.shape != self.values.shape:
            raise ValueError("相干性与干涉图尺寸不一致。")
        finite = self.coherence[np.isfinite(self.coherence)]
        if np.any((finite < 0) | (finite > 1)):
            raise ValueError("相干性必须在 0～1 之间。")
        if self.unit not in ("radian", "mm", "m"):
            raise ValueError("unit 只能是 radian、mm 或 m。")
        if self.phase_sign not in (-1, 1) or not np.isfinite(self.wavelength_m) or self.wavelength_m <= 0:
            raise ValueError("波长必须为正数，phase_sign 必须是 -1 或 1。")
        if self.mask is not None and np.shape(self.mask) != self.values.shape[1:]:
            raise ValueError("mask 尺寸与影像不一致。")
        if self.components is not None and np.shape(self.components) != self.values.shape:
            raise ValueError("解缠连通分量尺寸不一致。")
        return a

    def millimeters(self):
        if self.unit == "radian":
            return self.values * self.phase_sign * self.wavelength_m * 1000 / (4 * np.pi)
        return self.values * (1000 if self.unit == "m" else 1)


def invert(stack, threshold=0.35, reference=None, block_size=512, progress=None):
    a = stack.validate()
    if not np.isfinite(threshold) or not 0 <= threshold <= 1 or block_size < 1:
        raise ValueError("相干性阈值须为 0～1，block_size 须为正数。")
    m, h, w = stack.values.shape
    n = len(stack.dates)
    y = stack.millimeters().reshape(m, -1)
    coh = stack.coherence.reshape(m, -1)
    valid = np.isfinite(y) & np.isfinite(coh) & (coh >= threshold) & (coh > 0)
    if stack.mask is not None:
        valid &= np.asarray(stack.mask, bool).reshape(1, -1)
    components = None if stack.components is None else np.asarray(stack.components).reshape(m, -1)
    if components is not None:
        valid &= np.isfinite(components) & (components > 0)
    # One pixel must reference EVERY pair; selecting different reference pixels
    # in different pairs would introduce incompatible spatial reference frames.
    candidates = np.all(valid, axis=0)
    if reference is None:
        if not candidates.any():
            raise ValueError("没有对所有干涉图都有效的共同参考点，请调整阈值或清理干涉对。")
        score = np.min(np.where(valid, coh, -1), axis=0)
        ref_index = int(np.argmax(np.where(candidates, score, -1)))
        reference = divmod(ref_index, w)
    else:
        r, c = map(int, reference)
        if not (0 <= r < h and 0 <= c < w):
            raise ValueError("参考点超出影像范围。")
        ref_index = r * w + c
        reference = (r, c)
        if not candidates[ref_index]:
            raise ValueError("指定参考点在部分干涉图中无效或相干性不足。")
    y = y - y[:, ref_index:ref_index + 1]
    if components is not None:
        valid &= components == components[:, ref_index:ref_index + 1]
    ts = np.full((n, h * w), np.nan, np.float32)
    residual = np.full(h * w, np.nan, np.float32)
    temporal = np.full(h * w, np.nan, np.float32)
    count = valid.sum(axis=0).astype(np.int32)
    for start in range(0, h * w, block_size):
        stop = min(start + block_size, h * w)
        use = valid[:, start:stop]
        weights = np.where(use, coh[:, start:stop] ** 2, 0).T
        obs = np.where(use, y[:, start:stop], 0).T
        normal = np.einsum("mi,pm,mj->pij", a, weights, a, optimize=True)
        rhs = np.einsum("mi,pm,pm->pi", a, weights, obs, optimize=True)
        eig = np.linalg.eigvalsh(normal)
        # Reject local disconnected networks; regularization must not hide them.
        good = (eig[:, 0] > np.maximum(eig[:, -1] * 1e-10, 1e-12))
        ids = np.arange(start, stop)[good]
        if len(ids):
            x = np.linalg.solve(normal[good], rhs[good, :, None])[..., 0]
            ts[0, ids] = 0
            ts[1:, ids] = x.T
            err = obs[good] - x @ a.T
            wg = weights[good]
            residual[ids] = np.sqrt((wg * err**2).sum(axis=1) / wg.sum(axis=1))
            phase_error = err * (4 * np.pi) / (stack.wavelength_m * 1000)
            ug = use.T[good]
            temporal[ids] = np.abs((np.exp(1j * phase_error) * ug).sum(axis=1) / ug.sum(axis=1))
        if progress:
            progress(stop / (h * w), f"反演像素 {stop}/{h*w}")
    days = np.array([(datetime.strptime(d, "%Y%m%d") - datetime.strptime(stack.dates[0], "%Y%m%d")).days for d in stack.dates])
    years = days / 365.25
    centered = years - years.mean()
    velocity = np.sum(centered[:, None] * ts, axis=0) / np.sum(centered**2)
    return dict(dates=np.array(stack.dates), pairs=np.array(stack.pairs),
                displacement_mm=ts.reshape(n, h, w), velocity_mm_year=velocity.reshape(h, w),
                residual_mm=residual.reshape(h, w), temporal_coherence=temporal.reshape(h, w),
                valid_pair_count=count.reshape(h, w), mean_coherence=np.nanmean(stack.coherence, axis=0),
                reference_yx=np.array(reference), valid_mask=np.isfinite(velocity).reshape(h, w))


def select_pairs(dates, baselines_m=None, max_days=48, max_perp=150):
    normalized = [date_string(d) for d in dates]
    if len(set(normalized)) != len(normalized):
        raise ValueError("日期重复；同日多景需先拼接并按轨道分组。")
    if not np.isfinite(max_days) or max_days <= 0 or not np.isfinite(max_perp) or max_perp <= 0:
        raise ValueError("时间和空间基线阈值必须为正数。")
    dates = sorted(normalized)
    baseline = None if baselines_m is None else {date_string(k): float(v) for k, v in baselines_m.items()}
    if baseline is not None and (set(dates) - set(baseline) or not all(np.isfinite(v) for v in baseline.values())):
        raise ValueError("每个日期都需要有效的、相对同一主影像的垂直基线。")
    pairs = []
    for i, p in enumerate(dates):
        for q in dates[i + 1:]:
            dt = (datetime.strptime(q, "%Y%m%d") - datetime.strptime(p, "%Y%m%d")).days
            if dt <= max_days and (baseline is None or abs(baseline[q] - baseline[p]) <= max_perp):
                pairs.append((p, q))
    if not pairs:
        raise ValueError("没有满足基线阈值的干涉对。")
    design_matrix(dates, pairs)
    return pairs

