(function () {
  "use strict";

  const DEFAULT_LAYOUT = Object.freeze({
    offset_x_vw: 0,
    offset_y_vh: 0,
    scale: 1,
  });
  const MOVE_STEPS = Object.freeze({ small: 0.5, medium: 2, large: 5 });
  const LIMITS = Object.freeze({
    offset_x_vw: [-45, 45],
    offset_y_vh: [-35, 35],
    scale: [0.5, 2],
  });
  const SCALE_STEP = 0.05;

  function clamp(value, [minimum, maximum]) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function normalize(layout) {
    const result = {};
    for (const field of Object.keys(DEFAULT_LAYOUT)) {
      const value = Number(layout?.[field]);
      result[field] = Number.isFinite(value)
        ? clamp(value, LIMITS[field])
        : DEFAULT_LAYOUT[field];
    }
    return result;
  }

  function errorMessage(data, fallback) {
    return typeof data?.detail === "object"
      ? data.detail.message || fallback
      : data?.detail || fallback;
  }

  window.createPortraitLayoutController = function (options) {
    const {
      character,
      runtimeId,
      trigger,
      panel,
      scaleValue,
      status,
      resetButton,
      cancelButton,
      saveButton,
    } = options;
    const stepButtons = Array.from(
      panel.querySelectorAll("[data-portrait-step]"),
    );
    const moveButtons = Array.from(
      panel.querySelectorAll("[data-portrait-move]"),
    );
    const scaleButtons = Array.from(
      panel.querySelectorAll("[data-portrait-scale]"),
    );

    let savedLayout = { ...DEFAULT_LAYOUT };
    let draftLayout = { ...DEFAULT_LAYOUT };
    let moveStep = MOVE_STEPS.medium;
    let available = false;
    let saving = false;

    function apply(layout) {
      document.body.style.setProperty(
        "--portrait-offset-x",
        `${layout.offset_x_vw}vw`,
      );
      document.body.style.setProperty(
        "--portrait-offset-y",
        `${layout.offset_y_vh}vh`,
      );
      document.body.style.setProperty(
        "--portrait-scale",
        String(layout.scale),
      );
      scaleValue.textContent = `${Math.round(layout.scale * 100)}%`;
    }

    function setPanelOpen(open) {
      panel.hidden = !open;
      trigger.setAttribute("aria-pressed", String(open));
      document.body.classList.toggle("portrait-layout-open", open);
      if (!open) status.textContent = "";
    }

    function cancel() {
      draftLayout = { ...savedLayout };
      apply(savedLayout);
      setPanelOpen(false);
    }

    function setAvailable(enabled) {
      available = Boolean(enabled);
      trigger.disabled = !available || saving;
      if (!available && !panel.hidden) cancel();
    }

    function update(field, amount) {
      draftLayout[field] = Math.round(
        clamp(draftLayout[field] + amount, LIMITS[field]) * 100,
      ) / 100;
      apply(draftLayout);
    }

    async function load() {
      if (!runtimeId) return;
      try {
        const response = await fetch(
          `/api/characters/${character}/portrait-layout?runtime_id=` +
            encodeURIComponent(runtimeId),
          { cache: "no-store" },
        );
        const data = await response.json();
        if (!response.ok) {
          throw new Error(errorMessage(data, "无法读取立绘布局。"));
        }
        savedLayout = normalize(data);
        draftLayout = { ...savedLayout };
        apply(savedLayout);
      } catch (error) {
        console.warn("读取立绘布局失败，将使用默认位置。", error);
        apply(DEFAULT_LAYOUT);
      }
    }

    async function save() {
      if (saving) return;
      saving = true;
      status.textContent = "正在保存……";
      saveButton.disabled = true;
      trigger.disabled = true;
      try {
        const response = await fetch(
          `/api/characters/${character}/portrait-layout`,
          {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ runtime_id: runtimeId, ...draftLayout }),
          },
        );
        const data = await response.json();
        if (!response.ok) {
          throw new Error(errorMessage(data, "立绘布局保存失败。"));
        }
        savedLayout = normalize(data);
        draftLayout = { ...savedLayout };
        apply(savedLayout);
        setPanelOpen(false);
      } catch (error) {
        status.textContent = "保存失败：" + error.message;
      } finally {
        saving = false;
        saveButton.disabled = false;
        trigger.disabled = !available;
      }
    }

    trigger.setAttribute("aria-pressed", "false");
    trigger.addEventListener("click", () => {
      if (!available) return;
      if (!panel.hidden) {
        cancel();
        return;
      }
      draftLayout = { ...savedLayout };
      apply(draftLayout);
      setPanelOpen(true);
    });

    stepButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const selected = button.dataset.portraitStep;
        moveStep = MOVE_STEPS[selected] ?? MOVE_STEPS.medium;
        stepButtons.forEach((item) => {
          item.setAttribute("aria-pressed", String(item === button));
        });
      });
    });

    moveButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const changes = {
          up: ["offset_y_vh", -moveStep],
          down: ["offset_y_vh", moveStep],
          left: ["offset_x_vw", -moveStep],
          right: ["offset_x_vw", moveStep],
        };
        const change = changes[button.dataset.portraitMove];
        if (change) update(...change);
      });
    });

    scaleButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const amount = button.dataset.portraitScale === "increase"
          ? SCALE_STEP
          : -SCALE_STEP;
        update("scale", amount);
      });
    });

    resetButton.addEventListener("click", () => {
      draftLayout = { ...DEFAULT_LAYOUT };
      apply(draftLayout);
    });
    cancelButton.addEventListener("click", cancel);
    saveButton.addEventListener("click", save);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !panel.hidden) cancel();
    });

    apply(DEFAULT_LAYOUT);
    return { load, setAvailable };
  };
})();
