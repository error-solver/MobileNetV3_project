const navLinks = [...document.querySelectorAll('.nav-link')];
const sections = [...document.querySelectorAll('main section[id]')];
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });
document.querySelectorAll('.reveal').forEach((element) => revealObserver.observe(element));

const sectionObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      navLinks.forEach((link) => link.classList.toggle('active', link.hash === `#${entry.target.id}`));
    }
  });
}, { rootMargin: '-32% 0px -58% 0px' });
sections.forEach((section) => sectionObserver.observe(section));

const menuButton = document.querySelector('.menu-toggle');
const nav = document.querySelector('.desktop-nav');
menuButton.addEventListener('click', () => {
  const open = menuButton.getAttribute('aria-expanded') !== 'true';
  menuButton.setAttribute('aria-expanded', String(open));
  menuButton.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
  nav.classList.toggle('open', open);
});
navLinks.forEach((link) => link.addEventListener('click', () => {
  nav.classList.remove('open');
  menuButton.setAttribute('aria-expanded', 'false');
  menuButton.setAttribute('aria-label', 'Open menu');
}));

const input = document.getElementById('image-input');
const dropZone = document.getElementById('drop-zone');
const preview = document.getElementById('preview-image');
const empty = document.getElementById('upload-empty');
const badge = document.getElementById('preview-badge');
const fileName = document.getElementById('file-name');
const clear = document.getElementById('clear-image');
const message = document.getElementById('workspace-message');
let objectUrl = null;

function clearImage() {
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null;
  input.value = '';
  preview.removeAttribute('src');
  preview.hidden = true;
  badge.hidden = true;
  empty.hidden = false;
  clear.hidden = true;
  fileName.textContent = 'No image selected';
  message.textContent = 'To enable analysis, train a model and connect its checkpoint to an inference service. Image selection here stays in your browser.';
}
function showImage(file) {
  if (!file) return;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type) || file.size > 10 * 1024 * 1024) {
    message.textContent = 'Please choose a JPG, PNG or WEBP image under 10 MB.';
    return;
  }
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  preview.src = objectUrl;
  preview.hidden = false;
  badge.hidden = false;
  empty.hidden = true;
  clear.hidden = false;
  fileName.textContent = file.name;
  message.textContent = 'Image ready for local preview. Classification is unavailable until a trained checkpoint and inference service are connected.';
}
dropZone.addEventListener('click', () => input.click());
dropZone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); input.click(); }
});
input.addEventListener('change', () => showImage(input.files[0]));
['dragenter', 'dragover'].forEach((type) => dropZone.addEventListener(type, (event) => {
  event.preventDefault(); dropZone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach((type) => dropZone.addEventListener(type, (event) => {
  event.preventDefault(); dropZone.classList.remove('dragging');
}));
dropZone.addEventListener('drop', (event) => showImage(event.dataTransfer.files[0]));
clear.addEventListener('click', clearImage);
window.addEventListener('pagehide', () => { if (objectUrl) URL.revokeObjectURL(objectUrl); });
