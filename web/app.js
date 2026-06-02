const form = document.querySelector("#queryForm");
const statusBox = document.querySelector("#status");
const safetyText = document.querySelector("#safetyText");
const asrText = document.querySelector("#asrText");
const answerText = document.querySelector("#answerText");
const metricsBox = document.querySelector("#metrics");
const audioPlayer = document.querySelector("#audioPlayer");
const submitButton = form.querySelector('button[type="submit"]');
const startRecordButton = document.querySelector("#startRecord");
const stopRecordButton = document.querySelector("#stopRecord");
const recordState = document.querySelector("#recordState");
let playlist = [];
let playlistIndex = 0;
let audioContext = null;
let mediaStream = null;
let processor = null;
let source = null;
let recordedChunks = [];
let recordedBlob = null;
let recordingStartedAt = 0;

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

function mergeChunks(chunks) {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function resampleLinear(input, inputRate, outputRate) {
  if (inputRate === outputRate) return input;
  const ratio = inputRate / outputRate;
  const outputLength = Math.round(input.length / ratio);
  const output = new Float32Array(outputLength);

  for (let i = 0; i < outputLength; i += 1) {
    const position = i * ratio;
    const left = Math.floor(position);
    const right = Math.min(left + 1, input.length - 1);
    const weight = position - left;
    output[i] = input[left] * (1 - weight) + input[right] * weight;
  }

  return output;
}

function encodeWav(samples, sampleRate) {
  const bytesPerSample = 2;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);

  function writeString(offset, value) {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * bytesPerSample, true);

  let offset = 44;
  for (const sample of samples) {
    const clipped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true);
    offset += 2;
  }

  return new Blob([view], { type: "audio/wav" });
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("当前浏览器不支持麦克风录音");
    return;
  }

  recordedBlob = null;
  recordedChunks = [];
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
  audioContext = new AudioContext();
  source = audioContext.createMediaStreamSource(mediaStream);
  processor = audioContext.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = (event) => {
    recordedChunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  };
  source.connect(processor);
  processor.connect(audioContext.destination);
  recordingStartedAt = Date.now();
  startRecordButton.disabled = true;
  stopRecordButton.disabled = false;
  recordState.textContent = "录音中...";
}

async function stopRecording() {
  if (!audioContext || !processor) return;

  processor.disconnect();
  source.disconnect();
  mediaStream.getTracks().forEach((track) => track.stop());

  const inputRate = audioContext.sampleRate;
  await audioContext.close();
  audioContext = null;
  processor = null;
  source = null;
  mediaStream = null;

  const merged = mergeChunks(recordedChunks);
  const wavSamples = resampleLinear(merged, inputRate, 16000);
  recordedBlob = encodeWav(wavSamples, 16000);

  const seconds = ((Date.now() - recordingStartedAt) / 1000).toFixed(1);
  recordState.textContent = `已录音 ${seconds}s，可直接运行问答`;
  startRecordButton.disabled = false;
  stopRecordButton.disabled = true;
}

startRecordButton.addEventListener("click", async () => {
  try {
    await startRecording();
  } catch (error) {
    startRecordButton.disabled = false;
    stopRecordButton.disabled = true;
    recordState.textContent = `录音失败：${error.message}`;
  }
});

stopRecordButton.addEventListener("click", async () => {
  await stopRecording();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  setStatus("运行中，请稍等");
  metricsBox.innerHTML = "";

  try {
    if (audioContext && processor) {
      await stopRecording();
    }

    const data = new FormData();
    const formData = new FormData(form);
    const text = (formData.get("text") || "").trim();
    const file = form.audio.files[0];
    data.append("mode", formData.get("mode") || "streaming");
    if (text) {
      data.append("text", text);
    } else if (recordedBlob) {
      data.append("audio", recordedBlob, "recorded_question.wav");
    } else if (file) {
      data.append("audio", file, file.name);
    }

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
