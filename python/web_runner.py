"""Browser adapter for the original InSAR solver; no network or arbitrary code input."""
import csv
from datetime import datetime, timedelta, timezone
import gc
import io
import json
import math
import zipfile

import numpy as np
from core import Stack, invert, select_pairs

MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_EXPANDED_BYTES = 160 * 1024 * 1024
MAX_WORKING_BYTES = 256 * 1024 * 1024
MAX_EPOCHS = 36
MAX_PAIRS = 160
MAX_SIDE = 512
_state = None
_point = None


def reset():
    global _state, _point
    _state = None
    _point = None
    gc.collect()


def _dimensions(shapes):
    dates, pairs, values = shapes['dates'], shapes['pairs'], shapes['values']
    if len(dates) != 1 or not 3 <= dates[0] <= MAX_EPOCHS:
        raise ValueError('日期必须是一维数组，网页支持 3～36 个观测日期。')
    if len(pairs) != 2 or pairs[1] != 2 or not 2 <= pairs[0] <= MAX_PAIRS:
        raise ValueError('pairs 必须是 [干涉对数, 2]，最多支持 160 个干涉对。')
    if len(values) != 3 or values[0] != pairs[0] or not all(1 <= x <= MAX_SIDE for x in values[1:]):
        raise ValueError('values 必须是 [干涉对数, 行, 列]，行列均不能超过 512。')
    if shapes['coherence'] != values:
        raise ValueError('coherence 与 values 的尺寸必须一致。')
    if 'mask' in shapes and shapes['mask'] != values[1:]:
        raise ValueError('mask 必须是 [行, 列]。')
    if 'components' in shapes and shapes['components'] != values:
        raise ValueError('components 的尺寸必须与 values 一致。')
    m, h, w = values
    n = dates[0]
    estimate = m * h * w * 56 + n * h * w * 8 + h * w * 64 + 64 * n * n * 16
    if estimate > MAX_WORKING_BYTES:
        raise ValueError('该干涉栈超出网页内存预算（约 256 MiB 工作数组），请先裁剪研究区或使用桌面版。')
    if h * w * (n - 1) ** 3 > 450_000_000:
        raise ValueError('该干涉栈计算量超过网页上限，请裁剪研究区或减少日期数后重试。')


def load_npz(path):
    """Inspect every NPY header before allocating, then load with pickle disabled."""
    allowed = {'dates', 'pairs', 'values', 'coherence', 'unit', 'wavelength_m',
               'phase_sign', 'mask', 'metadata', 'components'}
    required = {'dates', 'pairs', 'values', 'coherence'}
    import os
    if os.path.getsize(path) > MAX_FILE_BYTES:
        raise ValueError('文件超过网页 25 MiB 上限，请先裁剪研究区。')
    shapes = {}
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > len(allowed) or sum(i.file_size for i in infos) > MAX_EXPANDED_BYTES:
                raise ValueError('NPZ 解压后超过 160 MiB，或包含过多数组。')
            for info in infos:
                if info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise ValueError('NPZ 包含不支持的加密或压缩方式。')
                if not info.filename.endswith('.npy'):
                    raise ValueError('NPZ 只能包含规范的 NumPy 数组。')
                key = info.filename[:-4]
                if key not in allowed or key in shapes:
                    raise ValueError('NPZ 存在未知或重复字段：' + key[:80])
                with archive.open(info) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=10000)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=10000)
                    else:
                        raise ValueError('请将 NPZ 数组保存为常规 NPY 1.0 或 2.0 格式。')
                    if dtype.hasobject or dtype.fields is not None:
                        raise ValueError('不支持 object、pickle 或结构化数组。')
                    if any(not isinstance(dim, int) or dim < 0 for dim in shape):
                        raise ValueError('NPZ 数组形状无效。')
                    size = math.prod(shape) * dtype.itemsize
                    if stream.tell() + size != info.file_size:
                        raise ValueError('NPZ 数组头声明的尺寸与文件长度不一致。')
                    if key in ('dates', 'pairs'):
                        if dtype.kind not in 'US' or dtype.itemsize > 64:
                            raise ValueError('dates 和 pairs 必须是 YYYYMMDD 字符串数组。')
                    elif key in ('unit', 'metadata'):
                        if shape != () or dtype.kind not in 'US' or size > 65536:
                            raise ValueError(key + ' 必须是较短的标量字符串。')
                    elif key in ('wavelength_m', 'phase_sign'):
                        if shape != () or dtype.kind not in 'fiu':
                            raise ValueError(key + ' 必须是数值标量。')
                    elif dtype.kind not in 'fiub':
                        raise ValueError(key + ' 必须是实数数组。')
                    shapes[key] = shape
            if not required <= shapes.keys():
                raise ValueError('NPZ 缺少必需字段：dates、pairs、values、coherence。')
            _dimensions(shapes)
        with np.load(path, allow_pickle=False, max_header_size=10000) as z:
            def scalar_text(key):
                value = z[key].item()
                return value.decode('utf-8') if isinstance(value, bytes) else str(value)
            metadata = json.loads(scalar_text('metadata')) if 'metadata' in z else {}
            if not isinstance(metadata, dict):
                raise ValueError('metadata 必须是 JSON 对象。')
            if 'synthetic' in metadata and not isinstance(metadata['synthetic'], bool):
                raise ValueError('metadata.synthetic 必须为 true 或 false。')
            mask = z['mask'] if 'mask' in z else None
            if mask is not None:
                mask = np.isfinite(mask) & (mask > 0)
            phase_sign = float(z['phase_sign']) if 'phase_sign' in z else -1
            if phase_sign not in (-1, 1):
                raise ValueError('phase_sign 必须为 -1 或 1。')
            stack = Stack(dates=z['dates'].tolist(), pairs=z['pairs'].tolist(),
                          values=z['values'], coherence=z['coherence'],
                          unit=scalar_text('unit') if 'unit' in z else 'radian',
                          wavelength_m=float(z['wavelength_m']) if 'wavelength_m' in z else 0.05546576,
                          phase_sign=int(phase_sign), mask=mask, metadata=metadata,
                          components=z['components'] if 'components' in z else None)
        stack.validate()
        return stack
    except (zipfile.BadZipFile, EOFError, OSError) as exc:
        raise ValueError('无法读取 NPZ，文件可能损坏或不是 NumPy 干涉栈。') from exc


def demo_stack():
    """Deterministic synthetic data derived from the desktop example."""
    rng = np.random.default_rng(42)
    offsets = [0, 12, 24, 36, 60, 72, 96, 108, 132, 156, 180, 204]
    dates = [(datetime(2024, 1, 1) + timedelta(days=d)).strftime('%Y%m%d') for d in offsets]
    pairs = select_pairs(dates, max_days=48)
    h, w = 96, 120
    yy, xx = np.mgrid[:h, :w]
    bowl = np.exp(-((xx - 68) ** 2 / 25 ** 2 + (yy - 52) ** 2 / 19 ** 2))
    velocity = -48 * bowl + 9 * np.exp(-((xx - 28) ** 2 + (yy - 32) ** 2) / 15 ** 2)
    t = np.array(offsets) / 365.25
    truth = t[:, None, None] * velocity + 2 * np.sin(2 * np.pi * t[:, None, None]) * bowl
    truth -= truth[:, 5:6, 5:6]
    wavelength = 0.05546576
    index = {d: i for i, d in enumerate(dates)}
    values = np.array([-(truth[index[q]] - truth[index[p]]) * 4 * np.pi / (wavelength * 1000)
                       for p, q in pairs])
    values += rng.normal(0, 0.025, values.shape)
    coherence = rng.uniform(0.65, 0.95, values.shape)
    coherence[:, 5, 5] = 0.99
    coherence[:, 72:84, 10:35] = 0.12
    values[:, :3] = np.nan
    metadata = {'synthetic': True, 'description': 'Synthetic demonstration; not satellite observations.',
                'seed': 42, 'coordinate_system': 'pixel', 'unwrapped': True}
    return Stack(dates, pairs, values, coherence, wavelength_m=wavelength, metadata=metadata)


def _flat(array):
    arr = np.asarray(array).reshape(-1)
    return [float(v) if np.isfinite(v) else None for v in arr]


def run(mode='demo', input_path='/input.npz', threshold=0.35, reference=None, progress=None):
    global _state, _point
    reset()
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('相干性阈值必须为 0～1。')
    if reference is not None:
        if len(reference) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                      or not np.isfinite(x) or int(x) != x for x in reference):
            raise ValueError('参考点须为 [行, 列] 两个整数，索引从 0 开始。')
        reference = [int(x) for x in reference]
    if progress:
        progress(25, '构造模拟干涉栈' if mode == 'demo' else '检查上传干涉栈')
    if mode == 'demo':
        stack = demo_stack()
    elif mode == 'file':
        stack = load_npz(input_path)
    else:
        raise ValueError('未知运行模式。')
    _dimensions({'dates': (len(stack.dates),), 'pairs': (len(stack.pairs), 2),
                 'values': stack.values.shape, 'coherence': stack.coherence.shape})
    def update(fraction, message):
        if progress:
            progress(30 + int(fraction * 64), message)
    result = invert(stack, threshold=threshold, reference=reference, block_size=64, progress=update)
    valid = result['valid_mask']
    velocities = result['velocity_mm_year'][valid]
    if not len(velocities):
        raise ValueError('没有可反演的有效像素，请检查相干性、干涉网络及参考点。')
    bounds = [float(velocities.min()), float(velocities.max())] if len(velocities) else [None, None]
    synthetic = stack.metadata.get('synthetic', False) is True
    report = {
        'method': 'weighted network least squares; coherence-squared heuristic weights',
        'engine': 'original insar_sbas/core.py running with Python / NumPy in the browser',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'synthetic': synthetic,
        'epochs': len(stack.dates), 'interferograms': len(stack.pairs),
        'reference_date': stack.dates[0], 'reference_yx': result['reference_yx'].tolist(),
        'reference_selection': 'user' if reference is not None else 'maximum minimum coherence',
        'threshold': threshold, 'input_unit': stack.unit, 'wavelength_m': stack.wavelength_m,
        'phase_sign': stack.phase_sign, 'phase_to_mm': 'phase_sign * wavelength_m * 1000 / (4*pi)',
        'units': {'displacement': 'mm', 'velocity': 'mm/year', 'residual': 'mm'},
        'los_positive': 'toward satellite only if the source phase sign convention matches phase_sign',
        'valid_pixels': int(valid.sum()), 'total_pixels': int(valid.size),
        'velocity_range_mm_year': bounds, 'coordinate_system': stack.metadata.get('coordinate_system', 'input grid; view uses pixel coordinates'),
        'limitations': ['Synthetic output is not an observation.' if synthetic else 'Source provenance and phase convention must be confirmed by the data owner.',
                       'Input must already be coregistered, unwrapped, and on a common pixel grid.',
                       'No SLC focusing, interferogram generation, unwrapping, atmospheric, orbit, or residual DEM correction.',
                       'Reference pixel stability is not established by high coherence.',
                       'Disconnected pixel networks are masked; quality metrics do not establish measurement accuracy.'],
    }
    h, w = valid.shape
    output = {'width': w, 'height': h, 'dates': stack.dates,
              'pairs': [list(p) for p in stack.pairs], 'reference': result['reference_yx'].tolist(),
              'velocity': _flat(result['velocity_mm_year']),
              'lastDisplacement': _flat(result['displacement_mm'][-1]),
              'temporalCoherence': _flat(result['temporal_coherence']),
              'residual': _flat(result['residual_mm']),
              'validPixels': int(valid.sum()), 'totalPixels': int(valid.size),
              'epochs': len(stack.dates), 'interferograms': len(stack.pairs),
              'synthetic': synthetic, 'velocityRange': bounds, 'report': report}
    # Publish the state only once all calculations and serialization succeed.
    payload = json.dumps(output, allow_nan=False, ensure_ascii=False)
    _state = {'stack': stack, 'result': result, 'report': report}
    _point = tuple(map(int, result['reference_yx']))
    if progress:
        progress(100, '反演完成')
    return payload


def point(row, col):
    global _point
    if _state is None:
        raise ValueError('请先完成一次反演。')
    if isinstance(row, bool) or isinstance(col, bool) or int(row) != row or int(col) != col:
        raise ValueError('像素坐标必须是整数。')
    row, col = int(row), int(col)
    result = _state['result']
    ts = result['displacement_mm']
    if not (0 <= row < ts.shape[1] and 0 <= col < ts.shape[2]):
        raise ValueError('像素坐标超出影像范围。')
    _point = (row, col)
    return json.dumps({'row': row, 'col': col, 'dates': result['dates'].tolist(),
                       'values': _flat(ts[:, row, col])}, allow_nan=False)


def export(format_name, output_path='/export.bin'):
    if _state is None:
        raise ValueError('当前没有可导出的结果，请先成功完成反演。')
    stack, result, report = (_state[k] for k in ('stack', 'result', 'report'))
    prefix = 'synthetic_' if report['synthetic'] else 'insar_'
    if format_name == 'json':
        data = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
        name, mime = prefix + 'report.json', 'application/json'
    elif format_name == 'csv':
        row, col = _point
        text = io.StringIO(newline='')
        writer = csv.writer(text)
        writer.writerow(['date', 'relative_los_displacement_mm', 'row', 'col', 'synthetic', 'reference_row', 'reference_col'])
        for date, value in zip(result['dates'], result['displacement_mm'][:, row, col]):
            writer.writerow([str(date), float(value) if np.isfinite(value) else '', row, col,
                             str(report['synthetic']).lower(), *result['reference_yx'].tolist()])
        data = text.getvalue().encode('utf-8-sig')
        name, mime = f'{prefix}point_r{row}_c{col}.csv', 'text/csv;charset=utf-8'
    elif format_name in ('npz', 'stack'):
        payload = io.BytesIO()
        if format_name == 'npz':
            np.savez_compressed(payload, **result, metadata=np.array(json.dumps(stack.metadata, ensure_ascii=False)),
                                report=np.array(json.dumps(report, ensure_ascii=False)))
            name = prefix + 'result.npz'
        else:
            optional = {}
            if stack.mask is not None:
                optional['mask'] = stack.mask
            if stack.components is not None:
                optional['components'] = stack.components
            np.savez_compressed(payload, dates=np.array(stack.dates), pairs=np.array(stack.pairs),
                                values=stack.values.astype(np.float32), coherence=stack.coherence.astype(np.float32),
                                unit=np.array(stack.unit), wavelength_m=stack.wavelength_m,
                                phase_sign=stack.phase_sign, metadata=np.array(json.dumps(stack.metadata, ensure_ascii=False)),
                                **optional)
            name = prefix + 'stack.npz'
        data = payload.getvalue()
        mime = 'application/octet-stream'
    else:
        raise ValueError('未知导出格式。')
    with open(output_path, 'wb') as output:
        output.write(data)
    return json.dumps({'filename': name, 'mime': mime})

