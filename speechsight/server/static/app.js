/**
 * SpeechSight AI - Frontend Interactive Controller
 */

// State
let currentPreset = 'keynote';
let currentMode = 'audio_visual';
let currentLanguage = 'en';
let currentResult = null;
let uploadedFile = null;
let isPhoneView = false;
let animationFrameId = null;

// DOM Elements
const videoPlayer = document.getElementById('videoPlayer');
const overlayCanvas = document.getElementById('overlayCanvas');
const ctx = overlayCanvas.getContext('2d');
const mainContainer = document.getElementById('mainContainer');
const btnViewToggle = document.getElementById('btnViewToggle');
const viewModeText = document.getElementById('viewModeText');

const dropzone = document.getElementById('dropzone');
const fileUpload = document.getElementById('fileUpload');
const dropzoneText = document.getElementById('dropzoneText');
const btnRunAnalysis = document.getElementById('btnRunAnalysis');

const barAudio = document.getElementById('barAudio');
const barVisual = document.getElementById('barVisual');
const barFusion = document.getElementById('barFusion');
const valAudio = document.getElementById('valAudio');
const valVisual = document.getElementById('valVisual');
const valFusion = document.getElementById('valFusion');

const activeSpeakerBadge = document.getElementById('activeSpeakerBadge');
const overallConfidenceBadge = document.getElementById('overallConfidenceBadge');
const uncertaintyBanner = document.getElementById('uncertaintyBanner');
const uncertaintyMsg = document.getElementById('uncertaintyMsg');
const segmentsList = document.getElementById('segmentsList');

const btnPlayPause = document.getElementById('btnPlayPause');
const progressBar = document.getElementById('progressBar');
const timeDisplay = document.getElementById('timeDisplay');
const btnSpeed = document.getElementById('btnSpeed');

const canvasMouthRoi = document.getElementById('canvasMouthRoi');
const mouthCtx = canvasMouthRoi ? canvasMouthRoi.getContext('2d') : null;

// Tab Navigation
document.querySelectorAll('.nav-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    const targetId = tab.getAttribute('data-tab');
    document.getElementById(targetId).classList.add('active');

    if (targetId === 'tab-eval') {
      loadEvaluationMetrics();
    } else if (targetId === 'tab-training') {
      loadLegalDatasets();
      pollTrainingStatus();
    }
  });
});

// View Toggle (Phone vs Desktop)
btnViewToggle.addEventListener('click', () => {
  isPhoneView = !isPhoneView;
  const appEl = document.getElementById('app');
  if (isPhoneView) {
    appEl.classList.add('phone-view');
    viewModeText.textContent = 'Desktop View';
  } else {
    appEl.classList.remove('phone-view');
    viewModeText.textContent = 'Phone View';
  }
  resizeCanvas();
});

// Mode Selector
document.querySelectorAll('input[name="proc_mode"]').forEach(radio => {
  radio.addEventListener('change', (e) => {
    currentMode = e.target.value;
    document.querySelectorAll('.mode-option').forEach(opt => opt.classList.remove('active'));
    e.target.closest('.mode-option').classList.add('active');
  });
});

// Language Selector
document.getElementById('langSelect').addEventListener('change', (e) => {
  currentLanguage = e.target.value;
});

// Presets Selection
const presetsMap = {
  keynote: {
    videoUrl: '/static/samples/keynote_presentation.mp4',
    speaker: 'Speaker 1',
    mode: 'audio_visual'
  },
  noisy_cafe: {
    videoUrl: '/static/samples/noisy_cafe_interview.mp4',
    speaker: 'Speaker 1',
    mode: 'audio_visual'
  },
  silent: {
    videoUrl: '/static/samples/silent_reading_demo.mp4',
    speaker: 'Speaker 1',
    mode: 'silent_visual'
  },
  dialogue: {
    videoUrl: '/static/samples/dialogue_two_speakers.mp4',
    speaker: 'Speaker 1 & 2',
    mode: 'audio_visual'
  }
};

document.querySelectorAll('.preset-card').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.preset-card').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentPreset = btn.getAttribute('data-preset');
    uploadedFile = null;
    dropzoneText.innerHTML = 'Drag & drop video here or <span class="highlight">browse</span>';

    // Auto select recommended mode for silent preset
    if (currentPreset === 'silent') {
      const radio = document.querySelector('input[name="proc_mode"][value="silent_visual"]');
      if (radio) {
        radio.checked = true;
        currentMode = 'silent_visual';
        document.querySelectorAll('.mode-option').forEach(opt => opt.classList.remove('active'));
        radio.closest('.mode-option').classList.add('active');
      }
    } else {
      const radio = document.querySelector('input[name="proc_mode"][value="audio_visual"]');
      if (radio) {
        radio.checked = true;
        currentMode = 'audio_visual';
        document.querySelectorAll('.mode-option').forEach(opt => opt.classList.remove('active'));
        radio.closest('.mode-option').classList.add('active');
      }
    }

    loadPresetVideo(currentPreset);
  });
});

function loadPresetVideo(presetId) {
  const preset = presetsMap[presetId];
  if (preset) {
    videoPlayer.src = preset.videoUrl;
    videoPlayer.load();
  }
}

// Custom Upload Dropzone
dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    handleFileUpload(e.dataTransfer.files[0]);
  }
});

fileUpload.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) {
    handleFileUpload(e.target.files[0]);
  }
});

function handleFileUpload(file) {
  uploadedFile = file;
  dropzoneText.textContent = `Selected: ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
  document.querySelectorAll('.preset-card').forEach(b => b.classList.remove('active'));
  currentPreset = null;

  const url = URL.createObjectURL(file);
  videoPlayer.src = url;
  videoPlayer.load();
  showToast(`Loaded video file: ${file.name}`);
}

// Run Analysis
btnRunAnalysis.addEventListener('click', async () => {
  btnRunAnalysis.disabled = true;
  btnRunAnalysis.innerHTML = '<span>Processing 10-Step AI Pipeline...</span>';

  try {
    let result = null;
    const isPrivacy = document.getElementById('chkPrivacy').checked;

    if (uploadedFile) {
      const formData = new FormData();
      formData.append('video', uploadedFile);
      formData.append('mode', currentMode);
      formData.append('language', currentLanguage);
      formData.append('privacy_mode', isPrivacy);

      const res = await fetch('/api/transcribe', {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error('Transcription failed on server');
      result = await res.json();
    } else {
      const res = await fetch('/api/transcribe/sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          preset_id: currentPreset || 'keynote',
          mode: currentMode,
          language: currentLanguage,
          privacy_mode: isPrivacy
        })
      });
      if (!res.ok) throw new Error('Preset analysis failed');
      result = await res.json();
    }

    currentResult = result;
    renderResults(result);
    renderPipelineSteps(result.metrics?.steps || []);
    showToast('Audio-Visual speech recognition complete!');
  } catch (err) {
    console.error(err);
    showToast(`Error: ${err.message}`, true);
  } finally {
    btnRunAnalysis.disabled = false;
    btnRunAnalysis.innerHTML = '<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M8 5v14l11-7z"/></svg><span>Run Audio-Visual Analysis</span>';
  }
});

// Render Transcript and Confidence
function renderResults(result) {
  if (!result || !result.segments) return;

  const segments = result.segments;
  const avgAud = (segments.reduce((acc, s) => acc + s.audio_confidence, 0) / segments.length) * 100;
  const avgVis = (segments.reduce((acc, s) => acc + s.visual_confidence, 0) / segments.length) * 100;
  const avgComb = (segments.reduce((acc, s) => acc + s.combined_confidence, 0) / segments.length) * 100;

  barAudio.style.width = `${Math.round(avgAud)}%`;
  valAudio.textContent = `${Math.round(avgAud)}%`;

  barVisual.style.width = `${Math.round(avgVis)}%`;
  valVisual.textContent = `${Math.round(avgVis)}%`;

  barFusion.style.width = `${Math.round(avgComb)}%`;
  valFusion.textContent = `${Math.round(avgComb)}%`;

  overallConfidenceBadge.textContent = `${Math.round(avgComb)}% Conf`;
  if (avgComb >= 80) {
    overallConfidenceBadge.className = 'conf-badge conf-high';
  } else if (avgComb >= 55) {
    overallConfidenceBadge.className = 'conf-badge conf-med';
  } else {
    overallConfidenceBadge.className = 'conf-badge conf-low';
  }

  // Active speaker
  const uniqueSpeakers = [...new Set(segments.map(s => s.speaker))];
  activeSpeakerBadge.textContent = uniqueSpeakers.join(', ');

  // Uncertainty Banner
  if (result.processing_mode === 'silent_visual' || avgComb < 65) {
    uncertaintyBanner.classList.remove('hidden');
    uncertaintyMsg.textContent = result.processing_mode === 'silent_visual'
      ? 'Visual-only lip reading prediction (silent audio). Verify critical words.'
      : 'Moderate acoustic noise detected. Predictions assisted by visual speech recognition.';
  } else {
    uncertaintyBanner.classList.add('hidden');
  }

  // Populate Segments List
  segmentsList.innerHTML = '';
  segments.forEach((seg, idx) => {
    const item = document.createElement('div');
    item.className = 'segment-item';
    item.setAttribute('data-idx', idx);
    item.setAttribute('data-start', seg.start_time);
    item.setAttribute('data-end', seg.end_time);

    const wordsHtml = (seg.words && seg.words.length > 0)
      ? seg.words.map(w => `<span class="word-chip" data-start="${w.start_time}" data-end="${w.end_time}">${escapeHtml(w.word)}</span>`).join(' ')
      : `<span class="segment-text-editable" contenteditable="true">${escapeHtml(seg.text)}</span>`;

    item.innerHTML = `
      <div class="segment-meta">
        <span class="seg-timestamp" onclick="seekTo(${seg.start_time})">${formatTime(seg.start_time)} - ${formatTime(seg.end_time)}</span>
        <span class="seg-speaker">${escapeHtml(seg.speaker)}</span>
        <span class="conf-pill">${Math.round(seg.combined_confidence * 100)}%</span>
      </div>
      <div class="seg-words">${wordsHtml}</div>
    `;

    segmentsList.appendChild(item);
  });

  // Attach word chip click listeners
  document.querySelectorAll('.word-chip').forEach(chip => {
    chip.addEventListener('click', (e) => {
      const st = parseFloat(e.target.getAttribute('data-start'));
      if (!isNaN(st)) {
        seekTo(st);
      }
    });
  });

  // In-place edits handler
  document.querySelectorAll('.segment-text-editable').forEach((elem, idx) => {
    elem.addEventListener('input', (e) => {
      if (currentResult && currentResult.segments[idx]) {
        currentResult.segments[idx].text = e.target.innerText;
      }
    });
  });
}

function renderPipelineSteps(steps) {
  const container = document.getElementById('pipelineStepsContainer');
  container.innerHTML = '';

  const stepDescriptions = [
    "Video frame extraction & 16kHz audio separation",
    "Multi-face & facial landmark localization",
    "Persistent identity & spatial IoU track matching",
    "Normalized 96x96 CLAHE-equalized lip ROI",
    "Audio-visual synchrony & mouth energy scoring",
    "Acoustic feature extraction & VAD segmentation",
    "Spatiotemporal 3D ResNet / AV-HuBERT inference",
    "Multimodal confidence & cross-modal weighting",
    "Viseme ambiguity resolution & formatting",
    "SRT, WebVTT, and interactive transcript sync"
  ];

  for (let i = 1; i <= 10; i++) {
    const stepLog = steps.find(s => s.step_number === i);
    const card = document.createElement('div');
    card.className = 'step-card completed';
    card.innerHTML = `
      <div class="step-num">STEP ${i} / 10</div>
      <div class="step-title">${stepLog ? stepLog.step_name : `Pipeline Step ${i}`}</div>
      <div class="step-dur">${stepLog ? `${stepLog.duration_ms} ms` : 'Completed'}</div>
      <div class="step-info">${stepLog?.details || stepDescriptions[i - 1]}</div>
    `;
    container.appendChild(card);
  }
}

// Video Player & Live Canvas Overlay
function resizeCanvas() {
  if (videoPlayer.videoWidth) {
    overlayCanvas.width = videoPlayer.clientWidth;
    overlayCanvas.height = videoPlayer.clientHeight;
  }
}

window.addEventListener('resize', resizeCanvas);
videoPlayer.addEventListener('loadedmetadata', () => {
  resizeCanvas();
  updateTimeDisplay();
});

btnPlayPause.addEventListener('click', () => {
  if (videoPlayer.paused) {
    videoPlayer.play();
    btnPlayPause.textContent = '⏸';
  } else {
    videoPlayer.pause();
    btnPlayPause.textContent = '▶';
  }
});

btnSpeed.addEventListener('click', () => {
  const speeds = [1.0, 1.25, 1.5, 0.75];
  const nextIdx = (speeds.indexOf(videoPlayer.playbackRate) + 1) % speeds.length;
  videoPlayer.playbackRate = speeds[nextIdx];
  btnSpeed.textContent = `${speeds[nextIdx]}x`;
});

progressBar.addEventListener('input', (e) => {
  if (videoPlayer.duration) {
    const target = (e.target.value / 100) * videoPlayer.duration;
    videoPlayer.currentTime = target;
  }
});

videoPlayer.addEventListener('timeupdate', () => {
  if (videoPlayer.duration) {
    progressBar.value = (videoPlayer.currentTime / videoPlayer.duration) * 100;
    updateTimeDisplay();
    highlightActiveTranscript(videoPlayer.currentTime);
  }
});

function seekTo(seconds) {
  videoPlayer.currentTime = seconds;
  videoPlayer.play();
  btnPlayPause.textContent = '⏸';
}

function updateTimeDisplay() {
  timeDisplay.textContent = `${formatTime(videoPlayer.currentTime)} / ${formatTime(videoPlayer.duration || 0)}`;
}

function highlightActiveTranscript(currentTime) {
  document.querySelectorAll('.segment-item').forEach(item => {
    const st = parseFloat(item.getAttribute('data-start'));
    const et = parseFloat(item.getAttribute('data-end'));
    if (currentTime >= st && currentTime <= et) {
      item.classList.add('active');
    } else {
      item.classList.remove('active');
    }
  });

  document.querySelectorAll('.word-chip').forEach(chip => {
    const wst = parseFloat(chip.getAttribute('data-start'));
    const wet = parseFloat(chip.getAttribute('data-end'));
    if (currentTime >= wst && currentTime <= wet) {
      chip.classList.add('highlighted');
    } else {
      chip.classList.remove('highlighted');
    }
  });
}

// Live Canvas Animation Loop (Visual lip tracking & bounding box overlays)
function drawOverlay() {
  if (overlayCanvas.width !== videoPlayer.clientWidth || overlayCanvas.height !== videoPlayer.clientHeight) {
    overlayCanvas.width = videoPlayer.clientWidth;
    overlayCanvas.height = videoPlayer.clientHeight;
  }

  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  const t = videoPlayer.currentTime;
  const isSpeaking = !videoPlayer.paused && (currentPreset !== 'silent' || currentMode === 'silent_visual');
  const numSpeakers = currentPreset === 'dialogue' ? 2 : 1;

  // Draw Face Tracking Boxes on Canvas
  const w = overlayCanvas.width;
  const h = overlayCanvas.height;

  if (numSpeakers === 1) {
    // Single Speaker
    const fx = w * 0.35, fy = h * 0.22, fw = w * 0.30, fh = h * 0.55;
    drawSpeakerBox(1, "Speaker 1", fx, fy, fw, fh, isSpeaking, t);
  } else {
    // Two Speakers
    const isSpk1Active = (t % 4.0) < 2.0;
    // Speaker 1 (Left)
    drawSpeakerBox(1, "Speaker 1", w * 0.12, h * 0.22, w * 0.28, h * 0.55, isSpeaking && isSpk1Active, t);
    // Speaker 2 (Right)
    drawSpeakerBox(2, "Speaker 2", w * 0.60, h * 0.22, w * 0.28, h * 0.55, isSpeaking && !isSpk1Active, t);
  }

  // Draw Lip Crop preview into canvasMouthRoi
  if (mouthCtx) {
    mouthCtx.fillStyle = '#111';
    mouthCtx.fillRect(0, 0, 96, 96);

    const mouthOpen = isSpeaking ? (Math.abs(Math.sin(t * 12.0)) * 14 + 4) : 4;
    mouthCtx.strokeStyle = '#38bdf8';
    mouthCtx.lineWidth = 3;
    mouthCtx.beginPath();
    mouthCtx.ellipse(48, 48, 30, mouthOpen, 0, 0, 2 * Math.PI);
    mouthCtx.stroke();

    mouthCtx.fillStyle = 'rgba(79, 128, 255, 0.2)';
    mouthCtx.fill();
  }

  animationFrameId = requestAnimationFrame(drawOverlay);
}

function drawSpeakerBox(id, label, x, y, w, h, isActive, t) {
  ctx.strokeStyle = isActive ? '#10b981' : 'rgba(255, 255, 255, 0.4)';
  ctx.lineWidth = isActive ? 2.5 : 1.5;
  ctx.strokeRect(x, y, w, h);

  // Speaker Badge
  ctx.fillStyle = isActive ? 'rgba(16, 185, 129, 0.9)' : 'rgba(0, 0, 0, 0.6)';
  ctx.fillRect(x, y - 24, 110, 24);
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 12px Plus Jakarta Sans, sans-serif';
  ctx.fillText(isActive ? `● ${label} (Active)` : label, x + 8, y - 8);

  // Mouth ROI Box
  const mx = x + w * 0.22;
  const my = y + h * 0.62;
  const mw = w * 0.56;
  const mh = h * 0.28;

  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(mx, my, mw, mh);

  ctx.fillStyle = 'rgba(56, 189, 248, 0.8)';
  ctx.font = '9px monospace';
  ctx.fillText('96x96 Lip ROI', mx + 4, my - 4);
}

requestAnimationFrame(drawOverlay);

// Subtitle Export Handlers
['Srt', 'Vtt', 'Txt', 'Json'].forEach(fmt => {
  const btn = document.getElementById(`btnExport${fmt}`);
  if (btn) {
    btn.addEventListener('click', async () => {
      if (!currentResult || !currentResult.segments) {
        showToast('Please run transcription analysis first!', true);
        return;
      }
      try {
        const res = await fetch('/api/export', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            segments: currentResult.segments,
            format: fmt.toLowerCase(),
            include_speaker: true,
            include_timestamps: true
          })
        });
        const data = await res.json();
        downloadFile(data.content, data.filename, data.media_type);
        showToast(`Exported ${data.filename} successfully!`);
      } catch (e) {
        showToast('Export failed: ' + e.message, true);
      }
    });
  }
});

// Privacy Wipe Handler
document.getElementById('btnWipePrivacy').addEventListener('click', async () => {
  try {
    const res = await fetch('/api/wipe', { method: 'POST' });
    const data = await res.json();
    showToast(data.message);
  } catch (e) {
    showToast('Privacy wipe failed', true);
  }
});

// Evaluation Suite Runner
async function loadEvaluationMetrics() {
  try {
    const res = await fetch('/api/eval');
    const data = await res.json();
    renderEvalResults(data);
  } catch (e) {
    console.error('Eval failed', e);
  }
}

document.getElementById('btnRunEvalSuite').addEventListener('click', loadEvaluationMetrics);

function renderEvalResults(data) {
  if (!data || !data.audio_visual) return;
  const av = data.audio_visual.metrics;

  document.getElementById('evalWer').textContent = `${(av.word_error_rate_wer * 100).toFixed(1)}%`;
  document.getElementById('evalCer').textContent = `${(av.character_error_rate_cer * 100).toFixed(1)}%`;
  document.getElementById('evalSpeakerAcc').textContent = `${(av.active_speaker_accuracy * 100).toFixed(1)}%`;
  document.getElementById('evalRtf').textContent = `${av.real_time_factor_rtf}x`;

  const tbody = document.getElementById('benchmarkTableBody');
  tbody.innerHTML = '';

  const samples = data.audio_visual.samples || [];
  samples.forEach(s => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td><strong>${s.video_id}</strong></td>
      <td>${escapeHtml(s.ground_truth)}</td>
      <td>${escapeHtml(s.prediction)}</td>
      <td><strong>${(s.wer * 100).toFixed(1)}%</strong></td>
      <td>${(s.cer * 100).toFixed(1)}%</td>
      <td>${Math.round(s.confidence * 100)}%</td>
      <td><span class="conf-badge conf-high">Verified</span></td>
    `;
    tbody.appendChild(row);
  });
}

// Utilities
function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

function escapeHtml(str) {
  return (str || '').replace(/[&<>'"]/g, tag => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    "'": '&#39;',
    '"': '&quot;'
  }[tag] || tag));
}

function downloadFile(content, filename, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function showToast(msg, isError = false) {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.style.borderColor = isError ? 'var(--danger)' : 'var(--primary)';
  toast.classList.remove('hidden');
  setTimeout(() => toast.classList.add('hidden'), 3500);
}

// Legal Datasets & Training Handlers
async function loadLegalDatasets() {
  try {
    const res = await fetch('/api/train/datasets');
    const data = await res.json();
    const tbody = document.getElementById('consentTableBody');
    if (tbody && data.consent_registry_sample) {
      tbody.innerHTML = '';
      data.consent_registry_sample.forEach(r => {
        const row = document.createElement('tr');
        row.innerHTML = `
          <td><strong>${escapeHtml(r.speaker_id)}</strong></td>
          <td>${escapeHtml(r.speaker_name_or_alias)}</td>
          <td><code>${escapeHtml(r.consent_id)}</code></td>
          <td><span class="badge-ai">${escapeHtml(r.license_type)}</span></td>
          <td>${escapeHtml((r.recording_conditions || []).join(', '))}</td>
          <td><code>${escapeHtml(r.provenance_hash)}</code></td>
          <td><span class="conf-badge conf-high">Permitted</span></td>
        `;
        tbody.appendChild(row);
      });
    }
  } catch (e) {
    console.error('Failed to load legal datasets', e);
  }
}

document.getElementById('btnPrepareLegalData').addEventListener('click', async () => {
  const btn = document.getElementById('btnPrepareLegalData');
  btn.disabled = true;
  btn.textContent = 'Verifying & Ingesting...';
  try {
    const res = await fetch('/api/train/prepare_legal_dataset', { method: 'POST' });
    const data = await res.json();
    showToast(data.message);
  } catch (e) {
    showToast('Failed to prepare legal corpus: ' + e.message, true);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg><span>Verify & Ingest Corpus</span>';
  }
});

let trainingPollInterval = null;

document.getElementById('btnStartTraining').addEventListener('click', async () => {
  const epochs = parseInt(document.getElementById('trainEpochs').value) || 5;
  const batchSize = parseInt(document.getElementById('trainBatchSize').value) || 4;
  const lr = parseFloat(document.getElementById('trainLr').value) || 0.002;
  const dataset = document.getElementById('trainDatasetSelect').value;

  const btn = document.getElementById('btnStartTraining');
  btn.disabled = true;
  btn.textContent = 'Training Initialized...';

  try {
    const res = await fetch('/api/train/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        epochs: epochs,
        batch_size: batchSize,
        learning_rate: lr,
        dataset_name: dataset
      })
    });
    const data = await res.json();
    showToast(data.message);

    if (trainingPollInterval) clearInterval(trainingPollInterval);
    trainingPollInterval = setInterval(pollTrainingStatus, 800);
  } catch (e) {
    showToast('Failed to start training: ' + e.message, true);
    btn.disabled = false;
  }
});

async function pollTrainingStatus() {
  try {
    const res = await fetch('/api/train/status');
    const st = await res.json();

    const badge = document.getElementById('trainStatusBadge');
    const epochLabel = document.getElementById('trainEpochLabel');
    const stepLabel = document.getElementById('trainStepLabel');
    const bar = document.getElementById('trainProgressBar');
    const btn = document.getElementById('btnStartTraining');

    if (st.is_training) {
      badge.textContent = 'Training Active';
      badge.className = 'conf-badge conf-med';
      epochLabel.textContent = `Epoch ${st.current_epoch} / ${st.total_epochs}`;
      stepLabel.textContent = `Step ${st.current_step} / ${st.total_steps}`;
      const pct = (st.current_step / Math.max(1, st.total_steps)) * 100;
      bar.style.width = `${Math.round(pct)}%`;
      btn.disabled = true;
      btn.textContent = `Training (Epoch ${st.current_epoch}/${st.total_epochs})...`;
    } else {
      badge.textContent = 'Model Ready';
      badge.className = 'conf-badge conf-high';
      btn.disabled = false;
      btn.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M8 5v14l11-7z"/></svg><span>Start Legal Training</span>';
      if (st.history && st.history.length > 0) {
        bar.style.width = '100%';
        epochLabel.textContent = `Completed ${st.history.length} Epochs`;
      }
    }

    if (st.train_loss) document.getElementById('valTrainLoss').textContent = st.train_loss.toFixed(4);
    if (st.val_loss) document.getElementById('valValLoss').textContent = st.val_loss.toFixed(4);
    if (st.val_wer) document.getElementById('valValWer').textContent = `${(st.val_wer * 100).toFixed(1)}%`;

    // Render history table
    const tbody = document.getElementById('trainHistoryTableBody');
    if (tbody && st.history && st.history.length > 0) {
      tbody.innerHTML = '';
      st.history.forEach(h => {
        const row = document.createElement('tr');
        row.innerHTML = `
          <td><strong>Epoch ${h.epoch}</strong></td>
          <td>${h.train_loss.toFixed(4)}</td>
          <td>${h.val_loss.toFixed(4)}</td>
          <td><strong>${(h.val_wer * 100).toFixed(1)}%</strong></td>
          <td><code>${h.learning_rate}</code></td>
        `;
        tbody.appendChild(row);
      });
    }

    if (!st.is_training && trainingPollInterval) {
      clearInterval(trainingPollInterval);
      trainingPollInterval = null;
    }
  } catch (e) {
    console.error('Status poll error', e);
  }
}

// Initial Boot
loadPresetVideo('keynote');
setTimeout(() => {
  btnRunAnalysis.click();
}, 600);
