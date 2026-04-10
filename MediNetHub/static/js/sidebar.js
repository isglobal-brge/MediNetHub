/**
 * MediNetHub Sidebar - Collapse/Expand logic
 * Persists state in localStorage.
 */
(function() {
  'use strict';

  var STORAGE_KEY = 'medinet-sidebar-expanded';

  function initSidebar() {
    var sidebar = document.getElementById('sidebar');
    var toggle = document.getElementById('sidebar-toggle');

    if (!sidebar || !toggle) return;

    // Restore state from localStorage
    var savedState = localStorage.getItem(STORAGE_KEY);
    if (savedState === 'true') {
      sidebar.classList.add('expanded');
    }

    // Toggle on click
    toggle.addEventListener('click', function() {
      sidebar.classList.toggle('expanded');
      var isExpanded = sidebar.classList.contains('expanded');
      localStorage.setItem(STORAGE_KEY, isExpanded);
    });
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initSidebar);
  } else {
    initSidebar();
  }
})();
