// Fixes primary sidebar toggle button across all viewport widths in
// sphinx-book-theme + pydata-sphinx-theme.
//
// Both pydata-sphinx-theme.js and sphinx-book-theme.js bind their click
// listeners to `document.querySelector(".primary-toggle")`, which matches the
// hidden `.bd-header .primary-toggle` rather than the visible
// `.bd-header-article .primary-toggle` button. Additionally,
// pydata-sphinx-theme switches from the modal drawer (`< 960px`) to the inline
// sidebar column at `960px`, whereas sphinx-book-theme only checks `992px`.
document.addEventListener('DOMContentLoaded', () => {
  const sidebar = document.getElementById('pst-primary-sidebar');
  const modal = document.getElementById('pst-primary-sidebar-modal');
  const headerToggle = document.querySelector('.bd-header .primary-toggle');
  const articleToggles = document.querySelectorAll(
      '.bd-header-article .primary-toggle'
  );

  if (!sidebar || !articleToggles.length) {
    return;
  }

  const desktopQuery = window.matchMedia('(min-width: 960px)');

  articleToggles.forEach((btn) => {
    btn.addEventListener(
        'click',
        (event) => {
          event.preventDefault();
          event.stopImmediatePropagation();
          btn.blur();

          if (desktopQuery.matches) {
            sidebar.classList.toggle('pst-sidebar-hidden');
          } else if (headerToggle) {
            sidebar.classList.remove('pst-sidebar-hidden');
            headerToggle.click();
          }
        },
        true
    );
  });

  // If the mobile modal dialog is open and the viewport is resized to desktop,
  // close the modal so pydata-sphinx-theme restores sidebar nodes inline.
  desktopQuery.addEventListener('change', (e) => {
    if (e.matches && modal && modal.open) {
      modal.close();
    }
  });
});
