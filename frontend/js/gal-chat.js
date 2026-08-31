(function () {
  "use strict";

  window.createGalChatView = function createGalChatView(options) {
    const {
      topbarArea,
      historyOverlay,
      historyList,
      openHistoryButton,
      closeHistoryButton,
      dialogueSpeaker,
      dialogueText,
    } = options;

    function setHistoryOpen(open) {
      const isGalMode = document.body.dataset.chatMode === "gal_chat";
      if (!isGalMode) {
        document.body.classList.remove("gal-history-open");
        historyOverlay.setAttribute("aria-hidden", "false");
        return;
      }
      const shouldOpen = Boolean(open && isGalMode);
      document.body.classList.toggle("gal-history-open", shouldOpen);
      historyOverlay.setAttribute("aria-hidden", String(!shouldOpen));
      if (shouldOpen) {
        requestAnimationFrame(() => {
          historyList.scrollTop = historyList.scrollHeight;
        });
      }
    }

    function setMode(mode) {
      if (mode === "text_chat") {
        document.body.classList.remove("gal-history-open");
        historyOverlay.setAttribute("aria-hidden", "false");
      } else {
        setHistoryOpen(false);
      }
    }

    function showDialogue(speaker, content) {
      dialogueSpeaker.textContent = speaker || "";
      dialogueText.textContent = content || "";
    }

    topbarArea.addEventListener("mouseenter", () => {
      topbarArea.classList.add("is-open");
    });
    topbarArea.addEventListener("mouseleave", () => {
      topbarArea.classList.remove("is-open");
    });
    openHistoryButton.addEventListener("click", (event) => {
      event.stopPropagation();
      setHistoryOpen(true);
    });
    closeHistoryButton.addEventListener("click", () => {
      setHistoryOpen(false);
    });
    historyOverlay.addEventListener("click", (event) => {
      if (event.target === historyOverlay) setHistoryOpen(false);
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") setHistoryOpen(false);
    });

    return {
      setHistoryOpen,
      setMode,
      showDialogue,
    };
  };
})();
