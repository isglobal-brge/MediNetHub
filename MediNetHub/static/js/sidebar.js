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

    var savedState = localStorage.getItem(STORAGE_KEY);
    if (savedState === 'true') {
      sidebar.classList.add('expanded');
    }

    toggle.addEventListener('click', function() {
      sidebar.classList.toggle('expanded');
      var isExpanded = sidebar.classList.contains('expanded');
      localStorage.setItem(STORAGE_KEY, isExpanded);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initSidebar);
  } else {
    initSidebar();
  }
})();
