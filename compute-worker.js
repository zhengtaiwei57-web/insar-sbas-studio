/* Run the original Python inversion in a dedicated worker. Pyodide is loaded from its public CDN. */
'use strict';
let runtimePromise = null;
let runtime = null;
let busy = false;
let hasResult = false;
const MAX_BYTES = 25 * 1024 * 1024;
const baseURL = new URL('.', self.location.href);
const PYODIDE_URL = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
const sendProgress = (percent, message) => self.postMessage({ type: 'progress', percent, message: String(message) });

async function initialize() {
  if (!runtimePromise) {
    runtimePromise = (async () => {
      sendProgress(3, '正在从公共 CDN 加载浏览器 Python 环境，首次加载约需下载 20 MB');
      importScripts(PYODIDE_URL + 'pyodide.js');
      const py = await loadPyodide({ indexURL: PYODIDE_URL });
      sendProgress(12, '正在加载 NumPy 科学计算组件');
      await py.loadPackage('numpy');
      const sources = await Promise.all(['core.py', 'web_runner.py'].map(async name => {
        const response = await fetch(new URL('python/' + name, baseURL));
        if (!response.ok) throw new Error('Python 程序文件加载失败：' + name);
        return [name, await response.text()];
      }));
      for (const [name, source] of sources) py.FS.writeFile('/home/pyodide/' + name, source);
      py.globals.set('_web_progress', sendProgress);
      py.runPython('import web_runner\nimport json');
      runtime = py;
      return py;
    })().catch(error => { runtimePromise = null; throw error; });
  }
  return runtimePromise;
}

function unlinkQuietly(py, path) {
  try { py.FS.unlink(path); } catch (_) { /* The file may not have been created. */ }
}

self.onmessage = async ({ data }) => {
  const request = data || {};
  if (busy) {
    self.postMessage({ type: 'error', message: '正在计算，请等待当前任务完成。' });
    return;
  }
  busy = true;
  const isRun = request.type === 'run';
  try {
    if (isRun) {
      hasResult = false;
      if (runtime) runtime.runPython('web_runner.reset()');
      if (!['demo', 'file'].includes(request.mode)) throw new Error('请选择模拟演示或 NPZ 干涉栈。');
      if (request.mode === 'file') {
        if (!(request.buffer instanceof ArrayBuffer) || request.buffer.byteLength === 0) throw new Error('请选择有效的 NPZ 文件。');
        if (request.buffer.byteLength > MAX_BYTES) throw new Error('文件超过网页 25 MiB 上限，请先裁剪研究区。');
        if (!String(request.filename || '').toLowerCase().endsWith('.npz')) throw new Error('网页支持 .npz 干涉栈，请按说明转换输入。');
      }
      const threshold = request.threshold ?? 0.35;
      if (typeof threshold !== 'number' || !Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw new Error('相干性阈值必须为 0～1。');
      const reference = request.reference ?? null;
      if (reference !== null && (!Array.isArray(reference) || reference.length !== 2 || reference.some(x => !Number.isInteger(x)))) throw new Error('参考点必须是从 0 开始的行、列整数。');
      const py = await initialize();
      if (request.mode === 'file') py.FS.writeFile('/input.npz', new Uint8Array(request.buffer));
      py.globals.set('_web_options', JSON.stringify({ mode: request.mode, threshold, reference }));
      let payload;
      try {
        payload = py.runPython('web_runner.run(**json.loads(_web_options), progress=_web_progress)');
      } finally {
        unlinkQuietly(py, '/input.npz');
        py.globals.delete('_web_options');
      }
      const output = JSON.parse(payload);
      hasResult = true;
      self.postMessage({ type: 'result', data: output });
    } else if (request.type === 'point') {
      if (!runtime || !hasResult) throw new Error('请先完成一次反演。');
      if (!Number.isInteger(request.row) || !Number.isInteger(request.col)) throw new Error('像素坐标必须是整数。');
      runtime.globals.set('_web_point', JSON.stringify([request.row, request.col]));
      try {
        const value = runtime.runPython('web_runner.point(*json.loads(_web_point))');
        self.postMessage({ type: 'point', ...JSON.parse(value) });
      } finally { runtime.globals.delete('_web_point'); }
    } else if (request.type === 'export') {
      if (!runtime || !hasResult) throw new Error('当前没有可导出的结果，请先成功完成反演。');
      if (!['npz', 'csv', 'json', 'stack'].includes(request.format)) throw new Error('不支持的导出格式。');
      runtime.globals.set('_web_export_format', request.format);
      try {
        const details = JSON.parse(runtime.runPython('web_runner.export(_web_export_format)'));
        const bytes = runtime.FS.readFile('/export.bin');
        // Copy out of the virtual filesystem before transferring ownership.
        const buffer = bytes.slice().buffer;
        self.postMessage({ type: 'download', ...details, buffer }, [buffer]);
      } finally {
        runtime.globals.delete('_web_export_format');
        unlinkQuietly(runtime, '/export.bin');
      }
    } else throw new Error('未知计算请求。');
  } catch (error) {
    if (isRun) {
      hasResult = false;
      if (runtime) {
        try { runtime.runPython('web_runner.reset()'); } catch (_) {}
        unlinkQuietly(runtime, '/input.npz');
      }
    }
    const raw = String(error?.message || error);
    const lines = raw.trim().split('\n');
    let message = lines[lines.length - 1].replace(/^(ValueError|RuntimeError|TypeError|Error):\s*/, '');
    if (/Failed to fetch|NetworkError|Load failed|importScripts/i.test(raw)) message = '计算环境下载失败，请检查网络后刷新重试。';
    self.postMessage({ type: 'error', message });
  } finally { busy = false; }
};

