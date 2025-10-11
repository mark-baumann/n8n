// Common UI helpers: toasts + debug

function debug(...args) {
  // eslint-disable-next-line no-console
  console.debug('[UI]', ...args);
}

function showToast(message, type = 'info', timeoutMs = 2500) {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    document.body.appendChild(container);
  }
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('show'));
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 200);
  }, timeoutMs);
}

// expose globally
window.UIHelpers = { debug, showToast };
