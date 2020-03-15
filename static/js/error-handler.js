function handleError (err, state) {
  console.error(err);

  if (err && err.error === 'User not built.') {
      alert('User model was not constructed properly. Restarting study.');
      window.location = '/';
  }

  stage = state;

  if (container) {
    container.classList.remove('loading');
    container.innerText = 'Failed to load. Please ask the supervisor for help.'
  }
}