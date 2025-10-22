(() => {
  const MODE_LABELS = {
    speed: "Speak Mode",
    balanced: "Quiet Mode",
    deep: "Silent Mode",
  };

  class CalendarPanel {
    constructor(root) {
      this.root = root;
      this.modeButtonRow = root.querySelector(".mode-buttons");
      this.timeline = root.querySelector(".calendar-timeline");
      this.nextBar = root.querySelector(".whats-next");
      this.actions = Array.from(root.querySelectorAll(".planner-actions button"));
    }

    setMode(mode) {
      if (!this.modeButtonRow) return;
      this.modeButtonRow.querySelectorAll("button").forEach((btn) => {
        const active = btn.dataset.mode === mode;
        btn.classList.toggle("active", active);
        btn.setAttribute("aria-pressed", active ? "true" : "false");
      });
      const label = MODE_LABELS[mode] || mode;
      this.root.querySelector(".mode-label").textContent = label;
    }

    updatePlan(plan) {
      if (!plan) return;
      this.setMode(plan.mode || "balanced");
      this._renderTimeline(plan.blocks || []);
      this._renderNext(plan.blocks || []);
    }

    _renderTimeline(blocks) {
      if (!this.timeline) return;
      this.timeline.innerHTML = "";
       this.actions.forEach((btn) => {
        btn.disabled = !blocks.length;
      });
      if (!blocks.length) {
        this.timeline.innerHTML = `<p class="empty">No events yet. Ask “Plan my day”.</p>`;
        return;
      }
      blocks.forEach((block) => {
        const el = document.createElement("div");
        el.className = `timeline-block ${block.type}`;
        el.innerHTML = `
          <div class="time">${formatTime(block.start)} – ${formatTime(block.end)}</div>
          <div class="title">${block.label}</div>
          <div class="meta">${formatMeta(block)}</div>
        `;
        this.timeline.appendChild(el);
      });
    }

    _renderNext(blocks) {
      if (!this.nextBar) return;
      const now = Date.now();
      const upcoming = blocks
        .map((block) => ({ ...block, startTs: Date.parse(block.start || "") }))
        .filter((block) => !Number.isNaN(block.startTs) && block.startTs >= now)
        .sort((a, b) => a.startTs - b.startTs)[0];
      if (!upcoming) {
        this.nextBar.textContent = "Nothing scheduled. Enjoy the open space.";
        return;
      }
      const start = formatTime(upcoming.start);
      const label = upcoming.label;
      this.nextBar.textContent = `${start} — ${label}`;
    }
  }

  function formatTime(value) {
    if (!value) return "--";
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return "--";
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function formatMeta(block) {
    if (block.type === "event" && block.meta?.location) return block.meta.location;
    if (block.type === "focus" && block.meta?.list) return `List: ${block.meta.list}`;
    if (block.type === "prep") return "Prep buffer";
    return "";
  }

  window.CalendarPanel = CalendarPanel;
})();
