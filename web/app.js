const form = document.querySelector("#queryForm");
const statusBox = document.querySelector("#status");
const safetyText = document.querySelector("#safetyText");
const asrText = document.querySelector("#asrText");
const answerText = document.querySelector("#answerText");
const metricsBox = document.querySelector("#metrics");
const audioPlayer = document.querySelector("#audioPlayer");
const submitButton = form.querySelector("button");
let playlist = [];
let playlistIndex = 0;

function setStatus(text) {
  statusBox.textContent = text;
}

function fmtMs(value) {
  if (value === null || value === undefined) return "-";
  return `${Math.round(value)} ms`;
}

function renderMetrics(latency = {}) {
  const labels = [
    ["asr_ms", "ASR"],
    ["safety_ms", "安全门控"],
    ["llm_ttft_ms", "LLM TTFT"],
    ["llm_ms", "LLM"],
    ["llm_total_ms", "LLM 总耗时"],
    ["tts_ms", "TTS"],
    ["tts_first_ms", "TTS 首句"],
    ["first_playable_ms", "首段可播放"],
    ["total_ms", "总耗时"],
  ];

  metricsBox.innerHTML = labels
    .filter(([key]) => key in latency)
    .map(
      ([key, label]) => `
        <div class="metric">
          <span>${label}</span>
          <strong>${fmtMs(latency[key])}</strong>
        </div>
      `
    )
    .join("");
}

function renderResult(data) {
  const safety = data.safety || {};
  safetyText.textContent = safety.blocked
    ? `拦截：${safety.reason || safety.category || "unsafe"}`
    : "通过：未命中不安全或越界输入";
  asrText.textContent = data.asr_text || "-";
  answerText.textContent = data.llm_text || "-";
  renderMetrics(data.latency || {});

  playlist = (data.audio_urls && data.audio_urls.length ? data.audio_urls : [data.audio_url]).filter(Boolean);
  playlistIndex = 0;

  if (playlist.length) {
    audioPlayer.src = `${playlist[0]}?t=${Date.now()}`;
    audioPlayer.hidden = false;
  } else {
    audioPlayer.removeAttribute("src");
  }
}

audioPlayer.addEventListener("ended", () => {
  if (playlistIndex + 1 >= playlist.length) return;
  playlistIndex += 1;
  audioPlayer.src = `${playlist[playlistIndex]}?t=${Date.now()}`;
  audioPlayer.play();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  setStatus("运行中，请稍等");
  metricsBox.innerHTML = "";

  try {
    const data = new FormData(form);
    const response = await fetch("/api/query", {
      method: "POST",
      body: data,
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "请求失败");
    }
    renderResult(result);
    setStatus(result.safety?.blocked ? "已由安全模块短路" : "运行完成");
  } catch (error) {
    setStatus(`错误：${error.message}`);
  } finally {
    submitButton.disabled = false;
  }
});
