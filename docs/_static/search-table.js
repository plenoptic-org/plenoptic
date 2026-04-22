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
    pageLength: 10,
    columnDefs: [
      {
        targets: 0,
        render: function(data, type, row, meta) {
          if (type === "display") {
            func_name = data.split("plenoptic")
            pre_tags = func_name[0].replace("<p>", "")
            post_tags = "</span></code>"
            func_name = "plenoptic" + func_name[1].split("<")[0]
            data_display = [func_name]
            func_name = func_name.split(".")
            i = func_name.length
            while (data_display[0].length >= 45) {
              i = i - 1;
              data_display = [func_name.slice(0, i).join(".") + ".", func_name.slice(i).join(".")];
            }
            data_display = data_display.map((d) => `${pre_tags}${d}${post_tags}`).join(' ')
            return `<p>${data_display}</p>`
          } else {
            return data
          }
        }
      }
    ]
  });
});
