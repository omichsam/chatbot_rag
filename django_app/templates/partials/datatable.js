// ---- Data tables: search, sorting and 10-row pages, so long lists never scroll forever ----
// <table data-table> pages its tbody rows and sorts by clicking a header (a cell's
// data-sort value wins over its text, for dates and sizes). <div data-paginate> pages
// its child elements the same way, without sorting. Options: data-page-size="10",
// data-label="documents" (used in "Search documents…"). Controls appear only when
// there is more than one page of items.
(function () {
  const SIZES = [10, 25, 50];
  const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });

  function sortValue(cell) {
    if (!cell) return "";
    if (cell.dataset.sort !== undefined) return cell.dataset.sort;
    return cell.textContent.replace(/\s+/g, " ").trim();
  }
  function compare(a, b) {
    const na = parseFloat(String(a).replace(/,/g, "")), nb = parseFloat(String(b).replace(/,/g, ""));
    const numeric = /^-?[\d,.]+(\s|%|$)/;
    if (numeric.test(a) && numeric.test(b) && !isNaN(na) && !isNaN(nb)) return na - nb;
    return collator.compare(String(a), String(b));
  }
  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function enhance(root) {
    const isTable = root.tagName === "TABLE";
    const body = isTable ? root.tBodies[0] : root;
    if (!body) return;
    const items = [...body.children].filter(n => !n.hasAttribute("data-table-skip"));
    // An empty-state row ("No documents yet") is not data.
    if (isTable && items.length === 1 && items[0].cells.length === 1) return;
    let size = Number(root.dataset.pageSize) || SIZES[0];
    if (items.length <= size) return;

    const label = root.dataset.label || (isTable ? "rows" : "items");
    const columns = isTable && root.tHead ? root.tHead.rows[0].cells.length : 1;
    let page = 1, query = "", sortCol = -1, sortDir = 1;
    const host = root.closest(".table-wrap") || root;

    // Toolbar: search + page size.
    const bar = el("div", "dt-bar");
    const search = el("input", "dt-search");
    search.type = "search";
    search.placeholder = `Search ${label}…`;
    search.setAttribute("aria-label", `Search ${label}`);
    const sizeLabel = el("label", "dt-size");
    const sizeSelect = el("select");
    SIZES.forEach(n => { const o = el("option", "", String(n)); o.value = n; o.selected = n === size; sizeSelect.appendChild(o); });
    sizeLabel.append("Show ", sizeSelect, " per page");
    bar.append(search, sizeLabel);
    host.before(bar);

    // Footer: "Showing 1–10 of 34" + pager.
    const foot = el("div", "dt-foot");
    const info = el("span", "dt-info");
    const pager = el("nav", "dt-pager");
    pager.setAttribute("aria-label", `${label} pages`);
    foot.append(info, pager);
    host.after(foot);

    const emptyRow = isTable ? el("tr", "dt-empty") : el("div", "dt-empty card");
    if (isTable) { const td = el("td", "", "No matching " + label); td.colSpan = columns; emptyRow.appendChild(td); }
    else emptyRow.textContent = "No matching " + label;

    // Sortable headers.
    if (isTable && root.tHead) {
      [...root.tHead.rows[0].cells].forEach((th, i) => {
        if (th.hasAttribute("data-nosort") || !th.textContent.trim()) return;
        th.classList.add("dt-sortable");
        th.tabIndex = 0;
        th.setAttribute("aria-sort", "none");
        const go = () => {
          sortDir = sortCol === i ? -sortDir : 1;
          sortCol = i;
          [...root.tHead.rows[0].cells].forEach(c => c.hasAttribute("aria-sort") && c.setAttribute("aria-sort", "none"));
          th.setAttribute("aria-sort", sortDir === 1 ? "ascending" : "descending");
          page = 1;
          render();
        };
        th.addEventListener("click", go);
        th.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
      });
    }

    function pageButton(text, target, current, disabled) {
      const b = el("button", "dt-page" + (current ? " on" : ""), text);
      b.type = "button";
      b.disabled = disabled;
      if (current) b.setAttribute("aria-current", "page");
      b.addEventListener("click", () => { page = target; render(); host.scrollIntoView({ block: "nearest" }); });
      return b;
    }

    function render() {
      const q = query.toLowerCase();
      let rows = q ? items.filter(r => r.textContent.toLowerCase().includes(q)) : items.slice();
      if (sortCol >= 0) rows.sort((a, b) => sortDir * compare(sortValue(a.cells[sortCol]), sortValue(b.cells[sortCol])));
      const pages = Math.max(1, Math.ceil(rows.length / size));
      page = Math.min(page, pages);
      const start = (page - 1) * size;
      const slice = rows.slice(start, start + size);
      body.replaceChildren(...(slice.length ? slice : [emptyRow]));

      const total = rows.length === items.length ? `${items.length}` : `${rows.length} (filtered from ${items.length})`;
      info.textContent = rows.length ? `Showing ${start + 1}–${start + slice.length} of ${total}` : `0 of ${items.length}`;

      pager.replaceChildren();
      if (pages > 1) {
        pager.appendChild(pageButton("‹", page - 1, false, page === 1)).setAttribute("aria-label", "Previous page");
        const shown = new Set([1, pages, page - 1, page, page + 1]);
        let last = 0;
        for (let p = 1; p <= pages; p++) {
          if (!shown.has(p)) continue;
          if (p - last > 1) pager.appendChild(el("span", "dt-gap", "…"));
          pager.appendChild(pageButton(String(p), p, p === page, false));
          last = p;
        }
        pager.appendChild(pageButton("›", page + 1, false, page === pages)).setAttribute("aria-label", "Next page");
      }
    }

    let timer;
    search.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(() => { query = search.value.trim(); page = 1; render(); }, 120); });
    sizeSelect.addEventListener("change", () => { size = Number(sizeSelect.value); page = 1; render(); });
    render();
  }

  document.querySelectorAll("table[data-table], [data-paginate]").forEach(enhance);
})();
