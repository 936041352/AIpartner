(function () {
  "use strict";

  window.createChatThemeController = function createChatThemeController(
    options,
  ) {
    const {
      character,
      runtimeId,
      picker,
      optionButtons,
      showError,
    } = options;
    const themes = new Set(
      optionButtons.map((button) => button.dataset.chatTheme),
    );
    let currentTheme = document.body.dataset.chatTheme || "default";
    let saving = false;

    function apply(theme) {
      const selected = themes.has(theme) ? theme : "default";
      currentTheme = selected;
      document.body.dataset.chatTheme = selected;
      optionButtons.forEach((button) => {
        button.setAttribute(
          "aria-pressed",
          String(button.dataset.chatTheme === selected),
        );
      });
    }

    function setSaving(value) {
      saving = value;
      optionButtons.forEach((button) => {
        button.disabled = value;
      });
    }

    async function select(theme) {
      if (saving || !themes.has(theme)) return;
      picker.open = false;
      if (theme === currentTheme) return;

      const previousTheme = currentTheme;
      apply(theme);
      setSaving(true);
      try {
        const response = await fetch(
          `/api/characters/${character}/chat-theme`,
          {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              runtime_id: runtimeId,
              chat_theme: theme,
            }),
          },
        );
        const data = await response.json();
        if (!response.ok) {
          const detail = typeof data.detail === "string"
            ? data.detail
            : data.detail?.message;
          throw new Error(detail || "聊天主题保存失败。");
        }
        apply(data.chat_theme);
      } catch (error) {
        apply(previousTheme);
        showError("聊天主题保存失败：" + error.message);
      } finally {
        setSaving(false);
      }
    }

    async function load() {
      try {
        const response = await fetch(
          `/api/characters/${character}/chat-theme?runtime_id=` +
            encodeURIComponent(runtimeId),
          { cache: "no-store" },
        );
        if (!response.ok) return;
        apply((await response.json()).chat_theme);
      } catch (error) {
        console.warn("读取聊天主题失败，将使用简洁白。", error);
      }
    }

    optionButtons.forEach((button) => {
      button.addEventListener("click", () => {
        select(button.dataset.chatTheme);
      });
    });
    document.addEventListener("click", (event) => {
      if (!picker.contains(event.target)) picker.open = false;
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") picker.open = false;
    });
    apply(currentTheme);

    return { load };
  };
})();
