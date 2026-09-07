const q = (selector) => document.querySelector(selector);
const runs = q('#runs');
const notice = q('#notice');

const taskLabel = {
  RUNNING: '运行中', COMPLETE: '已完成 · WBT 验收通过', PLANNED: '仅生成计划',
  WAITING: '等待启动', FAIL: '失败', TRAINING_QUEUED: '已批准 · 等待 GPU',
};
const stepLabel = {
  RUNNING: '运行中', COMPLETE: '已完成', PASS: 'PASS', WARN: '预览未生成',
  DRY_RUN: '仅生成计划', WAITING: '等待启动', FAIL: '失败',
};

const commonSteps = [
  ['01_gvhmr', '① GVHMR：视频 → 世界坐标人体动作'],
  ['02_smpl', '② SMPL：导出全局人体动作'],
];
const routeSteps = {
  gmr: [
    ['03_retarget', '③ GMR：人体动作 → G1 参考动作'],
    ['03_wbt', '④ WBT：参考动作 → WBT NPZ'],
    ['04_validation', '⑤ 自动验收：FPS、关节、四元数、刚体'],
    ['05_gmr_reference_preview', '⑥ GMR G1 参考动作回放'],
    ['06_wbt_policy_preview', '⑦ WBT G1 跟踪仿真'],
  ],
  protomotions: [
    ['03_proto_motion', '③ ProtoMotions：人体动作 → motion'],
    ['04_proto_pt', '④ ProtoMotions：motion → PT'],
    ['05_retarget', '⑤ PyRoki：PT → G1 参考动作'],
    ['03_wbt', '⑥ WBT：参考动作 → WBT NPZ'],
    ['04_validation', '⑦ 自动验收：FPS、关节、四元数、刚体'],
    ['05_protomotions_reference_preview', '⑧ PyRoki G1 参考动作回放'],
    ['06_protomotions_native_preview', '⑨ ProtoMotions 策略跟踪回放'],
    ['06_wbt_policy_preview', '⑩ WBT G1 跟踪仿真'],
  ],
};

function statusFor(id, stages) {
  const hit = stages.find((stage) => stage.path.startsWith(`${id}/`));
  return hit ? hit.status || 'RUNNING' : 'WAITING';
}
function esc(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
}
function results(name, stage) {
  window.open(`/results.html?run=${encodeURIComponent(name)}&stage=${encodeURIComponent(stage)}`, '_blank', 'noopener');
}
async function approve(name) {
  const response = await fetch(`/api/approve/${encodeURIComponent(name)}`, { method: 'POST' });
  const data = await response.json();
  notice.textContent = response.ok ? '已批准并进入训练队列；会在 GPU 空闲时自动训练。' : data.error;
  refresh();
}
async function refresh() {
  const data = await fetch('/api/runs').then((response) => response.json());
  runs.innerHTML = data.length ? '' : '<p class="empty">还没有任务。上传一个视频开始。</p>';
  data.forEach((run) => {
    const stages = [...commonSteps, ...(routeSteps[run.route] || routeSteps.gmr)];
    const cards = stages.map(([id, title]) => {
      const state = statusFor(id, run.stages);
      return `<div class="stage"><b>${title}</b><span class="stage-actions"><button class="secondary mini" onclick="results('${esc(run.name)}','${id}')">查看结果</button><span class="state state-${state.toLowerCase()}">${stepLabel[state] || state}</span></span></div>`;
    }).join('');
    const approved = run.status === 'COMPLETE';
    runs.insertAdjacentHTML('beforeend', `<article class="run"><div class="run-head"><h3>${esc(run.name)}</h3><span class="state state-${run.status.toLowerCase()}">${taskLabel[run.status] || run.status}</span></div>${approved ? `<div class="actions"><button onclick="approve('${esc(run.name)}')">批准进入训练队列</button></div>` : ''}<div class="stages">${cards}</div></article>`);
  });
}
q('#refresh').onclick = refresh;
q('#job').onsubmit = async (event) => {
  event.preventDefault();
  const file = q('#video').files[0];
  const name = q('#name').value;
  const route = document.querySelector('[name=route]:checked').value;
  notice.textContent = '正在上传视频…';
  const uploaded = await fetch('/api/upload', { method: 'POST', headers: { 'X-Filename': file.name, 'Content-Type': 'application/octet-stream' }, body: file });
  const upload = await uploaded.json();
  if (!uploaded.ok) { notice.textContent = upload.error; return; }
  notice.textContent = '正在启动 pipeline…';
  const started = await fetch('/api/build', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ video: upload.video, name, route }) });
  const result = await started.json();
  notice.textContent = started.ok ? `已启动：${result.name}` : result.error;
  refresh();
};
refresh();
// Dashboard updates do not need to compete with NoMachine or simulation I/O.
setInterval(() => { if (!document.hidden) refresh(); }, 15000);
