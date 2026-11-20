'use strict';
// The share choice lives in this browser (localStorage "avalon_share") and is prefilled in both modes.
function getShare() { try { return localStorage.getItem('avalon_share'); } catch (e) { return null; } }
function setShare(value) { try { localStorage.setItem('avalon_share', value); } catch (e) {} }

function render() {
  const share = getShare();
  document.querySelectorAll('[data-share]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.share === share)));
  const undecided = share !== 'yes' && share !== 'no';
  document.body.classList.toggle('undecided', undecided);
  document.querySelectorAll('[data-mode]').forEach(a => { a.setAttribute('aria-disabled', String(undecided)); a.tabIndex = undecided ? -1 : 0; });
}

document.querySelectorAll('[data-share]').forEach(b => b.addEventListener('click', () => { setShare(b.dataset.share); render(); }));
document.querySelectorAll('[data-mode]').forEach(a => a.addEventListener('click', e => { if (document.body.classList.contains('undecided')) e.preventDefault(); }));
render();
