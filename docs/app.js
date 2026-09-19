'use strict';

// Schematic samples illustrate the source-design choice; they are not data.
const svgNS = 'http://www.w3.org/2000/svg';
function normalPair(i) {
  const u = ((Math.sin((i + 1) * 12.9898) * 43758.5453) % 1 + 1) % 1;
  const v = ((Math.sin((i + 3) * 78.233) * 19341.3731) % 1 + 1) % 1;
  const radius = Math.sqrt(-2 * Math.log(Math.max(u, 0.015)));
  return [radius * Math.cos(2 * Math.PI * v), radius * Math.sin(2 * Math.PI * v)];
}
function renderSource(mode) {
  const learned = mode === 'learned';
  const points = document.querySelector('#source-points');
  const paths = document.querySelector('#transport-paths');
  points.replaceChildren(); paths.replaceChildren();
  for (let i = 0; i < 48; i++) {
    const [a, b] = normalPair(i);
    const x = Math.min(337, Math.max(24, (learned ? 231 : 126) + a * (learned ? 26 : 48)));
    const y = Math.min(289, Math.max(35, (learned ? 174 : 208) + b * (learned ? 25 : 47)));
    const point = document.createElementNS(svgNS, 'circle');
    point.setAttribute('cx', x); point.setAttribute('cy', y);
    point.setAttribute('r', i % 5 === 0 ? '3' : '2.1');
    point.setAttribute('opacity', String(0.35 + (i % 5) * 0.13));
    points.append(point);
    if (i % 4 === 0) {
      const path = document.createElementNS(svgNS, 'path');
      path.setAttribute('d', `M${x} ${y} Q${(x + 286) / 2 + 17} ${y - 53} 286 130`);
      paths.append(path);
    }
  }
  const glow = document.querySelector('#prior-glow');
  glow.setAttribute('cx', learned ? '229' : '126');
  glow.setAttribute('cy', learned ? '173' : '208');
  glow.setAttribute('rx', learned ? '83' : '115');
  glow.setAttribute('ry', learned ? '74' : '107');
  const label = document.querySelector('#source-label');
  label.textContent = learned ? 'Learned source' : 'Uninformed source';
  label.setAttribute('x', learned ? '174' : '72');
  label.setAttribute('y', learned ? '279' : '302');
  document.querySelector('#source-explanation').textContent = learned
    ? 'Start from a distribution that knows the robot’s state. Then refine with the generator.'
    : 'The same starting distribution for every state. The generator must transport samples toward relevant actions.';
  document.querySelectorAll('[data-source]').forEach(button => {
    const active = button.dataset.source === mode;
    button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active));
  });
}
document.querySelectorAll('[data-source]').forEach(button => button.addEventListener('click', () => renderSource(button.dataset.source)));
renderSource('learned');

const evaluations = {
  simulation: {
    scores: [['LeaP', 81.6, 0.4], ['VITA', 75.1, 0.9], ['A2A', 73.8, 2.2], ['BridgePolicy', 61.4, 1.3], ['NoPrior', 56.1, 2.5]],
    protocol: 'Mean ± standard deviation over 3 seeds. 15 tasks, 50 demonstrations per task, 100 evaluation episodes per task per seed. BridgePolicy is the authors’ reimplementation.',
    delta: '+25.5', title: 'Same architecture. A learned starting point.',
    text: 'NoPrior uses the same encoder, flow-matching generator and pipeline as LeaP, but starts from a standard Gaussian. LeaP improves average success from 56.1% to 81.6%.'
  },
  real: {
    scores: [['LeaP', 80.0], ['A2A', 68.3], ['VITA', 56.7], ['NoPrior', 46.7]],
    protocol: 'Average across Pick Cube, Close Box and Pick & Place Sandbags on Franka Research 3. 100 teleoperated demonstrations and 20 evaluation trials per task, with randomized object positions.',
    delta: '+33.3', title: 'From simulation to real manipulation.',
    text: 'LeaP reaches 80.0% average success across three real-world tasks, compared with 46.7% for NoPrior. The tasks cover rigid-object picking, articulated manipulation and deformable-object handling.'
  }
};
function renderResults(setting) {
  const data = evaluations[setting];
  const container = document.querySelector('#result-bars');
  container.replaceChildren();
  for (const [name, score, std] of data.scores) {
    const row = document.createElement('div');
    row.className = `bar-row${name === 'LeaP' ? ' featured' : ''}`;
    const label = document.createElement('span'); label.className = 'bar-label'; label.textContent = name;
    const track = document.createElement('div'); track.className = 'bar-track'; track.setAttribute('aria-hidden', 'true');
    const fill = document.createElement('div'); fill.className = 'bar-fill'; fill.style.setProperty('--score', `${score}%`); track.append(fill);
    const value = document.createElement('span'); value.className = 'bar-value'; value.textContent = `${score.toFixed(1)}%`;
    if (std !== undefined) { const uncertainty = document.createElement('small'); uncertainty.textContent = ` ± ${std.toFixed(1)}`; value.append(uncertainty); }
    row.append(label, track, value); container.append(row);
  }
  document.querySelector('#result-protocol').textContent = data.protocol;
  document.querySelector('#result-delta').replaceChildren(document.createTextNode(data.delta));
  const unit = document.createElement('span'); unit.textContent = ' pp'; document.querySelector('#result-delta').append(unit);
  document.querySelector('#insight-title').textContent = data.title;
  document.querySelector('#insight-text').textContent = data.text;
  document.querySelectorAll('[data-results]').forEach(button => {
    const active = button.dataset.results === setting;
    button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active));
  });
}
document.querySelectorAll('[data-results]').forEach(button => button.addEventListener('click', () => renderResults(button.dataset.results)));
renderResults('simulation');

document.querySelector('#copy-citation').addEventListener('click', async () => {
  const text = document.querySelector('#bibtex').textContent.trim();
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(text);
    status.textContent = 'BibTeX copied to clipboard.';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#bibtex'));
    selection.removeAllRanges(); selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});
