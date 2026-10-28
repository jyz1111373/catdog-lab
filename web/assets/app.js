/* 猫狗分类器前端 —— 原生 JS，无任何外部依赖（可完全离线运行） */
(function () {
  'use strict';

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const pct = (v) => (v * 100).toFixed(1) + '%';

  const state = {
    file: null,          // 当前待识别的图片（File 或 dataURL）
    isBlobUrl: false,
    sampleOffset: 0,
    training: false,
    pollTimer: null,
  };

  /* ---------------- 工具：二进制转 base64（分块，避免大图爆栈） ---------------- */
  function toBase64(bytes) {
    let out = '';
    const CHUNK = 0x8000;
    for (let i = 0; i < bytes.length; i += CHUNK) {
      out += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
    }
    return btoa(out);
  }

  /* ---------------- 网络 ---------------- */
  async function api(path, options = {}) {
    const res = await fetch(path, options);
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch (_) { data = text; }
    if (!res.ok) {
      const msg = (data && data.detail) || res.statusText || '请求失败';
      throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
    }
    return data;
  }

  /* ---------------- 页签 ---------------- */
  $$('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      $$('.tab').forEach((t) => t.classList.remove('active'));
      $$('.panel').forEach((p) => p.classList.remove('active'));
      tab.classList.add('active');
      $('#panel-' + tab.dataset.tab).classList.add('active');
      if (tab.dataset.tab === 'exp') loadExperiments();
      if (tab.dataset.tab === 'data') loadDataset();
    });
  });

  /* ---------------- 健康检查 / 模型 / 架构 ---------------- */
  async function loadHealth() {
    try {
      const h = await api('/api/health');
      $('#chip-device').textContent = `${h.device} · torch ${h.torch_version}`;
      $('#dot-health').classList.toggle('bad', h.status !== 'ok');
      if (h.active_model) {
        $('#chip-model').textContent = '当前模型 ' + h.active_model;
        $('#foot-model').textContent = '当前模型：' + h.active_model;
      } else {
        $('#chip-model').textContent = '尚未加载模型';
      }
      if (h.status !== 'ok') {
        showErr('#predict-err', '后端没有可用模型，请先到「训练」页训练，或运行 scripts/run_all.py');
      }
    } catch (e) {
      $('#chip-model').textContent = '服务未连接';
      $('#dot-health').classList.add('bad');
      showErr('#predict-err', '无法连接后端：' + e.message);
    }
  }

  async function loadArchs() {
    try {
      const d = await api('/api/archs');
      const sel = $('#f-arch');
      sel.innerHTML = '';
      d.arch_info.forEach((a) => {
        const o = document.createElement('option');
        o.value = a.name;
        o.textContent = `${a.name}  (${a.params})`;
        o.dataset.desc = a.desc;
        sel.appendChild(o);
      });
      sel.value = 'resnet18';
      updateArchDesc();
    } catch (_) { /* 后端未启动时静默 */ }
  }

  function updateArchDesc() {
    const o = $('#f-arch').selectedOptions[0];
    $('#f-arch-desc').textContent = o ? o.dataset.desc || '' : '';
  }
  $('#f-arch').addEventListener('change', updateArchDesc);

  /* ---------------- 识别页：选择 / 拖放 / 预览 ---------------- */
  const drop = $('#drop');
  const fileInput = $('#file');

  drop.addEventListener('click', (e) => { if (e.target !== fileInput) fileInput.click(); });
  fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });

  ['dragenter', 'dragover'].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) setFile(f);
  });

  function setFile(f) {
    state.file = f;
    const img = $('#preview');
    img.src = URL.createObjectURL(f);
    img.hidden = false;
    $('#dropInner').hidden = true;
    $('#btn-predict').disabled = false;
    hideErr('#predict-err');
  }

  $('#btn-clear').addEventListener('click', () => {
    state.file = null;
    $('#preview').hidden = true;
    $('#preview').src = '';
    $('#dropInner').hidden = false;
    $('#btn-predict').disabled = true;
    $('#result').hidden = true;
    $('#result-empty').hidden = false;
    fileInput.value = '';
  });

  $('#btn-predict').addEventListener('click', async () => {
    if (!state.file) return;
    const fd = new FormData();
    fd.append('file', state.file);
    await doPredict(
      () => api('/api/predict', { method: 'POST', body: fd }),
      state.file.name || null);
  });

  async function predictUrl(url, truth) {
    await doPredict(() => fetch('/api/dataset/image/' + encodeURIComponent(url))
      .then((r) => r.blob())
      .then(async (blob) => {
        const buf = new Uint8Array(await blob.arrayBuffer());
        const b64 = toBase64(buf);
        return api('/api/predict/base64', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: b64, truth }),
      });
    }), url, truth);
  }

  async function doPredict(call, name, truth) {
    hideErr('#predict-err');
    $('#btn-predict').disabled = true;
    try {
      const r = await call();
      renderResult(r, name, truth);
    } catch (e) {
      showErr('#predict-err', e.message);
    } finally {
      $('#btn-predict').disabled = !state.file;
    }
  }

  function renderResult(r, name, truth) {
    $('#result-empty').hidden = true;
    $('#result').hidden = false;

    $('#verdict-emoji').textContent = r.emoji;
    $('#verdict-label').textContent = `${r.label_cn}（${r.label}）`;
    $('#verdict-sub').textContent = `模型判定 · 架构 ${r.arch}`;
    $('#conf').textContent = pct(r.confidence);

    $('#bars').innerHTML = r.all.map((d) => `
      <div class="bar">
        <div class="bar-top"><span>${d.emoji} ${d.label_cn}</span><b>${pct(d.prob)}</b></div>
        <div class="bar-track"><div class="bar-fill ${d.name}" style="width:${(d.prob * 100).toFixed(2)}%"></div></div>
      </div>`).join('');

    $('#result-meta').innerHTML = `
      <div>推理耗时 <b>${r.inference_ms.toFixed(0)} ms</b></div>
      <div>权重文件 <b>${r.checkpoint}</b></div>
      ${name ? `<div style="grid-column:1/-1">图片 <b>${name}</b></div>` : ''}`;

    const t = r.truth || truth;
    const box = $('#truth-box');
    if (t) {
      const ok = t === r.label;
      box.hidden = false;
      box.className = 'truth ' + (ok ? 'ok' : 'bad');
      box.textContent = ok
        ? `✓ 预测正确（真实标签：${t}）`
        : `✗ 预测错误（真实标签：${t}，模型判为 ${r.label}）`;
    } else {
      box.hidden = true;
    }
  }

  /* ---------------- 拍照 ---------------- */
  let stream = null;
  $('#btn-camera').addEventListener('click', async () => {
    const box = $('#camera-box');
    if (!box.hidden) { stopCamera(); return; }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      $('#video').srcObject = stream;
      box.hidden = false;
    } catch (e) {
      showErr('#predict-err', '无法打开摄像头：' + e.message + '（需 localhost 或 HTTPS，且浏览器已授权）');
    }
  });
  $('#btn-cam-close').addEventListener('click', stopCamera);
  function stopCamera() {
    if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
    $('#camera-box').hidden = true;
  }
  $('#btn-shot').addEventListener('click', async () => {
    const v = $('#video'), c = $('#canvas');
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    const dataUrl = c.toDataURL('image/jpeg', 0.9);
    $('#preview').src = dataUrl; $('#preview').hidden = false;
    $('#dropInner').hidden = true;
    await doPredict(() => api('/api/predict/base64', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl }),
    }), '摄像头拍摄', null);
  });

  /* ---------------- 样本库 ---------------- */
  async function loadSamples() {
    try {
      const s = await api('/api/dataset/stats');
      const items = s.items;
      const start = state.sampleOffset % items.length;
      const pick = [];
      for (let i = 0; i < 6; i++) pick.push(items[(start + i) % items.length]);
      $('#sample-thumbs').innerHTML = pick.map((it) =>
        `<img src="${it.url}" title="${it.name}" data-name="${it.name}" data-label="${it.label}" loading="lazy" />`).join('');
      $$('#sample-thumbs img').forEach((img) =>
        img.addEventListener('click', () => {
          $('#preview').src = img.src; $('#preview').hidden = false;
          $('#dropInner').hidden = true;
          predictUrl(img.dataset.name, img.dataset.label);
        }));
    } catch (_) { /* 忽略 */ }
  }
  $('#btn-shuffle').addEventListener('click', () => { state.sampleOffset += 6; loadSamples(); });

  /* ---------------- 数据集页 ---------------- */
  async function loadDataset() {
    try {
      const s = await api('/api/dataset/stats');
      $('#data-stat').textContent =
        `共 ${s.total} 张 · 猫 ${s.counts.cat} 张 · 狗 ${s.counts.dog} 张`;
      $('#data-grid').innerHTML = s.items.map((it) =>
        `<img src="${it.url}" title="${it.name}（${it.label}）" data-name="${it.name}" data-label="${it.label}" loading="lazy" />`).join('');
      $$('#data-grid img').forEach((img) =>
        img.addEventListener('click', () => {
          $$('.tab').find((t) => t.dataset.tab === 'predict').click();
          $('#preview').src = img.src; $('#preview').hidden = false;
          $('#dropInner').hidden = true;
          predictUrl(img.dataset.name, img.dataset.label);
        }));
    } catch (e) { /* 忽略 */ }
  }

  /* ---------------- 训练 ---------------- */
  $('#btn-train').addEventListener('click', async () => {
    if (state.training) return;
    hideErr('#train-err');
    const payload = {
      arch: $('#f-arch').value,
      epochs: +$('#f-epochs').value,
      batch_size: +$('#f-batch').value,
      lr: +$('#f-lr').value,
      pretrained: $('#f-pretrained').checked,
      freeze: $('#f-freeze').checked,
      augment: $('#f-augment').checked,
      name: $('#f-name').value.trim() || null,
      note: 'Web 页面发起',
    };
    try {
      await api('/api/train', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      state.training = true;
      $('#btn-train').disabled = true;
      $('#progress').hidden = false;
      $('#train-result').hidden = true;
      $('#charts').innerHTML = '';
      state.pollTimer = setInterval(pollTrain, 700);
      pollTrain();
    } catch (e) { showErr('#train-err', e.message); }
  });

  async function pollTrain() {
    let st;
    try { st = await api('/api/train/status'); } catch (_) { return; }

    const total = st.progress.total_epochs || 1;
    const cur = st.progress.epoch || 0;
    $('#pbar-fill').style.width = (cur / total * 100).toFixed(1) + '%';
    $('#pstat-text').textContent = st.message || '训练中…';
    $('#pstat-epoch').textContent = `第 ${cur}/${total} 轮 · 已用 ${st.elapsed.toFixed(0)}s`;

    if (st.history && st.history.length) renderCharts(st.history);

    if (st.status === 'done' || st.status === 'error' || st.status === 'cancelled') {
      clearInterval(state.pollTimer);
      state.training = false;
      $('#btn-train').disabled = false;
      if (st.status === 'done') {
        $('#pstat-text').textContent = '✅ ' + st.message;
        renderTrainResult(st.result);
        loadHealth();          // 新模型已被自动设为当前模型
        loadExperiments();
      } else {
        showErr('#train-err', st.message || '训练失败');
      }
    }
  }

  function renderTrainResult(rec) {
    if (!rec) return;
    const m = rec.metrics || {};
    $('#train-result').hidden = false;
    $('#train-result').innerHTML = `
      <div class="m"><span>准确率</span><b>${pct(m.accuracy || 0)}</b></div>
      <div class="m"><span>精确率</span><b>${pct(m.precision_macro || 0)}</b></div>
      <div class="m"><span>召回率</span><b>${pct(m.recall_macro || 0)}</b></div>
      <div class="m"><span>F1</span><b>${pct(m.f1_macro || 0)}</b></div>`;
  }

  /* ---------------- 曲线图（SVG，无依赖） ---------------- */
  function renderCharts(history) {
    const loss = history.map((h) => h.train_loss);
    const valLoss = history.map((h) => h.val_loss);
    const acc = history.map((h) => h.accuracy);
    $('#charts').innerHTML =
      lineChart('训练 / 验证 Loss', [
        { name: 'train loss', color: '#4f46e5', values: loss },
        { name: 'val loss', color: '#f59e0b', values: valLoss },
      ]) +
      lineChart('验证集准确率', [
        { name: 'accuracy', color: '#10b981', values: acc },
      ], true);
  }

  function lineChart(title, series, isPct) {
    const W = 620, H = 190, PL = 46, PR = 14, PT = 22, PB = 26;
    const all = series.flatMap((s) => s.values).filter((v) => typeof v === 'number');
    if (!all.length) return '';
    let min = Math.min(...all), max = Math.max(...all);
    if (max - min < 1e-6) { max += 0.1; min -= 0.1; }
    const pad = (max - min) * 0.12; min -= pad; max += pad;
    if (isPct) { min = Math.max(0, Math.min(min, 0.95)); max = Math.min(1.02, Math.max(max, 1)); }

    const n = Math.max(...series.map((s) => s.values.length));
    const x = (i) => PL + (n <= 1 ? 0 : (i / (n - 1)) * (W - PL - PR));
    const y = (v) => PT + (1 - (v - min) / (max - min)) * (H - PT - PB);
    const fmt = (v) => Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(isPct ? 2 : 3);

    const grid = [0, 0.25, 0.5, 0.75, 1].map((t) => {
      const v = min + t * (max - min), yy = y(v);
      return `<line x1="${PL}" y1="${yy}" x2="${W - PR}" y2="${yy}" stroke="#eef0f4"/>
              <text x="${PL - 8}" y="${yy + 4}" font-size="11" fill="#9ca3af" text-anchor="end">${fmt(v)}</text>`;
    }).join('');

    const xlab = series[0].values.map((_, i) =>
      (n <= 8 || i % Math.ceil(n / 8) === 0 || i === n - 1)
        ? `<text x="${x(i)}" y="${H - 6}" font-size="11" fill="#9ca3af" text-anchor="middle">${i + 1}</text>`
        : '').join('');

    const paths = series.map((s) => {
      const pts = s.values.map((v, i) => `${x(i)},${y(v)}`).join(' ');
      const dots = s.values.map((v, i) =>
        `<circle cx="${x(i)}" cy="${y(v)}" r="2.6" fill="${s.color}"/>`).join('');
      return `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="2.2"
                stroke-linejoin="round" stroke-linecap="round"/>${dots}`;
    }).join('');

    const legend = `<div class="legend">${series.map((s) =>
      `<span><i style="background:${s.color}"></i>${s.name}</span>`).join('')}</div>`;

    return `<div style="margin-bottom:10px"><div class="small muted" style="margin-bottom:2px">${title}</div>
      <svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${grid}${paths}${xlab}</svg>
      ${legend}</div>`;
  }

  /* ---------------- 实验对比 ---------------- */
  async function loadExperiments() {
    let d;
    try { d = await api('/api/experiments'); } catch (_) { return; }
    const recs = (d.records || []).slice().sort((a, b) =>
      (b.metrics.accuracy || 0) - (a.metrics.accuracy || 0));

    // 条形图
    $('#exp-chart').innerHTML = recs.length ? recs.map((r) => {
      const a = r.metrics.accuracy || 0;
      const cls = a >= 0.9 ? 'high' : a >= 0.75 ? 'mid' : 'low';
      return `<div class="bar">
        <div class="bar-top"><span>${r.name}</span><b>${pct(a)}</b></div>
        <div class="bar-track"><div class="bar-fill ${cls}" style="width:${(a * 100).toFixed(1)}%;
          background:${a >= 0.9 ? 'linear-gradient(90deg,#34d399,#10b981)' :
                        a >= 0.75 ? 'linear-gradient(90deg,#fbbf24,#f59e0b)' :
                                    'linear-gradient(90deg,#f87171,#ef4444)'}"></div></div>
      </div>`;
    }).join('') : '<p class="muted">还没有实验记录，先去「训练」页跑一次。</p>';

    // 表格
    $('#exp-table tbody').innerHTML = recs.map((r) => {
      const m = r.metrics || {};
      const acc = m.accuracy || 0;
      const cls = acc >= 0.9 ? 'high' : acc >= 0.75 ? 'mid' : 'low';
      return `<tr>
        <td><b>${r.name}</b></td>
        <td>${r.arch}</td>
        <td class="num"><span class="badge ${cls}">${pct(acc)}</span></td>
        <td class="num">${pct(m.precision_macro || 0)}</td>
        <td class="num">${pct(m.recall_macro || 0)}</td>
        <td class="num">${pct(m.f1_macro || 0)}</td>
        <td class="num">${(r.train_seconds || 0).toFixed(1)}s</td>
        <td class="small muted" style="white-space:normal">${r.note || ''}</td>
      </tr>`;
    }).join('');
  }
  $('#btn-refresh-exp').addEventListener('click', loadExperiments);

  /* ---------------- 工具 ---------------- */
  function showErr(sel, msg) { const el = $(sel); el.textContent = msg; el.hidden = false; }
  function hideErr(sel) { $(sel).hidden = true; }

  /* ---------------- 启动 ---------------- */
  loadHealth();
  loadArchs();
  loadSamples();
  loadExperiments();
  // 若服务刚启动就有任务在跑（例如刷新页面），继续显示进度
  api('/api/train/status').then((st) => {
    if (st.running) {
      state.training = true;
      $('#btn-train').disabled = true;
      $('#progress').hidden = false;
      state.pollTimer = setInterval(pollTrain, 700);
    }
  }).catch(() => {});
})();
