/**
 * This script is for initializing the search table on the API index page. See
 * DataTables documentation for more information: https://datatables.net/
 *
 * Copied and modified from scikit-learn
 */

document.addEventListener("DOMContentLoaded", function () {
  new DataTable("table.search-table", {
    order: [], // Keep original order
    lengthMenu: [10, 25, { label: "All", value: -1 }],
    pageLength: 10
  });
});
