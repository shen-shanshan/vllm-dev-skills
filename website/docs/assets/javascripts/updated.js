// Mark tutorials whose "最后更新" date is more than STALE_DAYS old. The age is
// computed in the browser, so pages turn stale without a rebuild.
const STALE_DAYS = 90;

function markStale() {
  const now = new Date();
  document.querySelectorAll("[data-updated]").forEach((el) => {
    const updated = el.dataset.updated;
    if (!updated || el.dataset.checked) return;
    el.dataset.checked = "1";
    const days = Math.floor((now - new Date(updated + "T00:00:00")) / 86400000);
    if (days <= STALE_DAYS) return;
    el.classList.add("stale");
    if (el.classList.contains("doc-updated")) {
      const note = document.createElement("div");
      note.className = "stale-note";
      note.textContent =
        `本文已超过 3 个月（${days} 天）未更新，vLLM 的代码可能已经发生变化，内容仅供参考，请以最新代码为准。`;
      el.appendChild(note);
    } else {
      el.title = `已超过 3 个月（${days} 天）未更新`;
    }
  });
}

if (typeof document$ !== "undefined") {
  document$.subscribe(markStale);
} else {
  document.addEventListener("DOMContentLoaded", markStale);
}
