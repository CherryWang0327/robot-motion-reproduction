const params = new URLSearchParams(location.search);
const run = params.get('run');
const stage = params.get('stage');
const title = document.querySelector('#title');
const content = document.querySelector('#content');

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
}
async function main() {
  title.textContent = `${run} · ${stage}`;
  const response = await fetch(`/api/results/${encodeURIComponent(run)}?stage=${encodeURIComponent(stage)}`, { cache: 'no-store' });
  const data = await response.json();
  if (!response.ok) { content.textContent = data.error; return; }
  const videos = data.files.filter((file) => /\.(mp4|webm)$/i.test(file));
  const others = data.files.filter((file) => !/\.(mp4|webm)$/i.test(file));
  const videoHtml = videos.length
    ? `<h2>视频结果</h2><div class="video-grid">${videos.map((file) => `<figure class="video-card"><video controls playsinline preload="none" src="/api/file?path=${encodeURIComponent(file)}"></video><figcaption>${escapeHtml(file.split('/').pop())}</figcaption></figure>`).join('')}</div>`
    : '<p class="empty">该阶段尚未产生可预览视频。</p>';
  const filesHtml = others.map((file) => `<li><a target="_blank" href="/api/file?path=${encodeURIComponent(file)}">${escapeHtml(file.split('/').pop())}</a></li>`).join('') || '<li>暂无其他文件</li>';
  content.innerHTML = `${videoHtml}<h2>文件</h2><ul>${filesHtml}</ul>`;
}
main().catch((error) => { content.textContent = `结果页加载失败：${error.message}`; });
