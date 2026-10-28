/**
 * DeepVoice Shield — Frontend Application
 * =========================================
 * Handles direct PCM WAV microphone recording, waveform visualization,
 * API communication, and animated results display.
 */

// ==========================================
// Configuration
// ==========================================
// Supports same-origin (Vercel fullstack) or external backend URL
const API_BASE = (window.DEEPVOICE_API_URL || localStorage.getItem('deepvoice_api_url') || window.location.origin).replace(/\/+$/, '');
const MAX_RECORD_SECONDS = 30;

// ==========================================
// DOM Elements
// ==========================================
const tabRecord = document.getElementById('tabRecord');
const tabUpload = document.getElementById('tabUpload');
const recordModeContainer = document.getElementById('recordModeContainer');
const uploadModeContainer = document.getElementById('uploadModeContainer');
const audioDropzone = document.getElementById('audioDropzone');
const audioFileInput = document.getElementById('audioFileInput');
const selectedFileInfo = document.getElementById('selectedFileInfo');
const selectedFileName = document.getElementById('selectedFileName');
const btnAnalyzeFile = document.getElementById('btnAnalyzeFile');

const micButton = document.getElementById('micButton');
const micIcon = document.getElementById('micIcon');
const stopIcon = document.getElementById('stopIcon');
const micRipple = document.getElementById('micRipple');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const timerDisplay = document.getElementById('timerDisplay');
const waveformCanvas = document.getElementById('waveformCanvas');
const waveformContainer = document.getElementById('waveformContainer');
const analyzingState = document.getElementById('analyzingState');
const recordCard = document.getElementById('recordCard');
const resultsCard = document.getElementById('resultsCard');
const resetBtn = document.getElementById('resetBtn');
const heroWaveCanvas = document.getElementById('heroWaveCanvas');

// Result elements
const gaugeFill = document.getElementById('gaugeFill');
const gaugeValue = document.getElementById('gaugeValue');
const verdictLabel = document.getElementById('verdictLabel');
const verdictDesc = document.getElementById('verdictDesc');
const realBar = document.getElementById('realBar');
const fakeBar = document.getElementById('fakeBar');
const realValue = document.getElementById('realValue');
const fakeValue = document.getElementById('fakeValue');
const featureGrid = document.getElementById('featureGrid');

let selectedAudioFile = null;

// ==========================================
// State
// ==========================================
let audioContext = null;
let analyserNode = null;
let sourceNode = null;
let scriptProcessor = null;
let stream = null;
let recordedSamples = [];
let isRecording = false;
let recordInterval = null;
let recordSeconds = 0;
let waveformAnimId = null;

// ==========================================
// Particle Background
// ==========================================
(function initParticles() {
    const canvas = document.getElementById('particleCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let particles = [];
    const PARTICLE_COUNT = 60;

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    class Particle {
        constructor() {
            this.reset();
        }
        reset() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.size = Math.random() * 2 + 0.5;
            this.speedX = (Math.random() - 0.5) * 0.3;
            this.speedY = (Math.random() - 0.5) * 0.3;
            this.opacity = Math.random() * 0.3 + 0.05;
            this.hue = Math.random() > 0.5 ? 185 : 270; // cyan or purple
        }
        update() {
            this.x += this.speedX;
            this.y += this.speedY;
            if (this.x < 0 || this.x > canvas.width || this.y < 0 || this.y > canvas.height) {
                this.reset();
            }
        }
        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${this.hue}, 100%, 60%, ${this.opacity})`;
            ctx.fill();
        }
    }

    for (let i = 0; i < PARTICLE_COUNT; i++) {
        particles.push(new Particle());
    }

    function drawLines() {
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 150) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(0, 240, 255, ${0.03 * (1 - dist / 150)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {
            p.update();
            p.draw();
        });
        drawLines();
        requestAnimationFrame(animate);
    }
    animate();
})();

// ==========================================
// Hero Wave Animation
// ==========================================
(function initHeroWave() {
    if (!heroWaveCanvas) return;
    const ctx = heroWaveCanvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    function resize() {
        const rect = heroWaveCanvas.parentElement.getBoundingClientRect();
        heroWaveCanvas.width = rect.width * dpr;
        heroWaveCanvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);
    }
    resize();
    window.addEventListener('resize', resize);

    let t = 0;
    function draw() {
        const w = heroWaveCanvas.width / dpr;
        const h = heroWaveCanvas.height / dpr;
        ctx.clearRect(0, 0, w, h);

        // Draw multiple wave layers
        const waves = [
            { amp: 30, freq: 0.015, speed: 0.03, color: 'rgba(0, 240, 255, 0.3)', width: 2 },
            { amp: 20, freq: 0.02, speed: 0.025, color: 'rgba(124, 58, 237, 0.25)', width: 2 },
            { amp: 15, freq: 0.025, speed: 0.02, color: 'rgba(16, 185, 129, 0.2)', width: 1.5 },
            { amp: 10, freq: 0.03, speed: 0.035, color: 'rgba(0, 240, 255, 0.15)', width: 1 },
        ];

        waves.forEach(wave => {
            ctx.beginPath();
            ctx.moveTo(0, h / 2);
            for (let x = 0; x <= w; x++) {
                const y = h / 2 + Math.sin(x * wave.freq + t * wave.speed) * wave.amp
                    + Math.sin(x * wave.freq * 2.5 + t * wave.speed * 1.3) * wave.amp * 0.3;
                ctx.lineTo(x, y);
            }
            ctx.strokeStyle = wave.color;
            ctx.lineWidth = wave.width;
            ctx.stroke();
        });

        t++;
        requestAnimationFrame(draw);
    }
    draw();
})();

// ==========================================
// Waveform Visualization
// ==========================================
function drawWaveform() {
    if (!analyserNode || !waveformCanvas) return;
    const ctx = waveformCanvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = waveformContainer.getBoundingClientRect();
    waveformCanvas.width = rect.width * dpr;
    waveformCanvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const bufferLength = analyserNode.fftSize;
    const dataArray = new Float32Array(bufferLength);

    function draw() {
        waveformAnimId = requestAnimationFrame(draw);
        analyserNode.getFloatTimeDomainData(dataArray);

        ctx.clearRect(0, 0, w, h);

        // Background grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
        ctx.lineWidth = 1;
        for (let y = 0; y < h; y += 20) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }

        // Center line
        ctx.strokeStyle = 'rgba(0, 240, 255, 0.1)';
        ctx.beginPath();
        ctx.moveTo(0, h / 2);
        ctx.lineTo(w, h / 2);
        ctx.stroke();

        // Waveform
        const gradient = ctx.createLinearGradient(0, 0, w, 0);
        gradient.addColorStop(0, '#00f0ff');
        gradient.addColorStop(0.5, '#7c3aed');
        gradient.addColorStop(1, '#00f0ff');

        ctx.beginPath();
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 2;
        ctx.shadowColor = '#00f0ff';
        ctx.shadowBlur = 8;

        const sliceWidth = w / bufferLength;
        let x = 0;
        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i];
            const y = (v + 1) / 2 * h;
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
            x += sliceWidth;
        }
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Glow fill under waveform
        ctx.lineTo(w, h / 2);
        ctx.lineTo(0, h / 2);
        ctx.closePath();
        const fillGradient = ctx.createLinearGradient(0, 0, 0, h);
        fillGradient.addColorStop(0, 'rgba(0, 240, 255, 0.05)');
        fillGradient.addColorStop(1, 'rgba(124, 58, 237, 0.02)');
        ctx.fillStyle = fillGradient;
        ctx.fill();
    }
    draw();
}

function drawIdleWaveform() {
    if (!waveformCanvas) return;
    const ctx = waveformCanvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = waveformContainer.getBoundingClientRect();
    waveformCanvas.width = rect.width * dpr;
    waveformCanvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    let t = 0;

    function draw() {
        if (isRecording) return;
        waveformAnimId = requestAnimationFrame(draw);
        ctx.clearRect(0, 0, w, h);

        // Subtle idle animation
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(0, 240, 255, 0.15)';
        ctx.lineWidth = 1.5;
        for (let x = 0; x < w; x++) {
            const y = h / 2 + Math.sin(x * 0.02 + t * 0.02) * 5 + Math.sin(x * 0.01 + t * 0.01) * 3;
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        t++;
    }
    draw();
}

// Start idle waveform on load
drawIdleWaveform();

// ==========================================
// WAV Encoder (16-bit PCM Mono)
// ==========================================
function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    function writeString(offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    // RIFF header
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');

    // fmt subchunk
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);          // SubChunk1Size (16 for PCM)
    view.setUint16(20, 1, true);           // AudioFormat (1 = PCM)
    view.setUint16(22, 1, true);           // NumChannels (1 = Mono)
    view.setUint32(24, sampleRate, true);  // SampleRate
    view.setUint32(28, sampleRate * 2, true); // ByteRate (SampleRate * NumChannels * BitsPerSample/8)
    view.setUint16(32, 2, true);           // BlockAlign (NumChannels * BitsPerSample/8)
    view.setUint16(34, 16, true);          // BitsPerSample (16-bit)

    // data subchunk
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    // PCM 16-bit audio data
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([view], { type: 'audio/wav' });
}

// ==========================================
// Downsample Helper to 16kHz
// ==========================================
function downsampleBuffer(buffer, sourceRate, targetRate = 16000) {
    if (sourceRate === targetRate) return buffer;
    const ratio = sourceRate / targetRate;
    const newLength = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;
    while (offsetResult < result.length) {
        const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
        let accum = 0, count = 0;
        for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
            accum += buffer[i];
            count++;
        }
        result[offsetResult] = count > 0 ? accum / count : 0;
        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
    }
    return result;
}

// ==========================================
// Recording Logic
// ==========================================
micButton.addEventListener('click', toggleRecording);

async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    // 1. Check if browser environment supports mediaDevices (requires localhost or HTTPS)
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
            const localhostUrl = `http://localhost:${window.location.port || 5000}`;
            statusText.innerHTML = `⚠️ Microphone blocked on IP address. <a href="${localhostUrl}" style="color: #00f0ff; text-decoration: underline; font-weight: bold;">Click to switch to http://localhost:${window.location.port || 5000}</a>`;
            statusIndicator.className = 'status-indicator';
            alert(`⚠️ Browser Security Restriction:\n\nModern browsers (Chrome, Edge, Safari) only permit microphone access on 'localhost' or 'HTTPS'.\n\nYou opened: ${window.location.href}\n\nPlease click OK to switch to: ${localhostUrl}`);
            window.location.href = localhostUrl;
            return;
        } else {
            statusText.textContent = 'Microphone API not supported or disabled in this browser.';
            statusIndicator.className = 'status-indicator';
            alert('Your browser does not support or has blocked the microphone recording API. Please test with a modern browser or use the "Upload Audio File" tab.');
            return;
        }
    }

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false,
                channelCount: 1,
            }
        });

        // Setup Web Audio Context
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        audioContext = new AudioCtx();
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }

        analyserNode = audioContext.createAnalyser();
        analyserNode.fftSize = 2048;

        sourceNode = audioContext.createMediaStreamSource(stream);
        sourceNode.connect(analyserNode);

        // Setup ScriptProcessor to capture raw PCM Float32 samples
        recordedSamples = [];
        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        scriptProcessor.onaudioprocess = (e) => {
            if (!isRecording) return;
            const channelData = e.inputBuffer.getChannelData(0);
            recordedSamples.push(new Float32Array(channelData));
        };

        sourceNode.connect(scriptProcessor);
        // Connect through silent gain node to prevent speaker feedback loop into microphone
        const muteGain = audioContext.createGain();
        muteGain.gain.value = 0;
        scriptProcessor.connect(muteGain);
        muteGain.connect(audioContext.destination);

        isRecording = true;
        recordSeconds = 0;
        updateUI('recording');

        // Live waveform
        cancelAnimationFrame(waveformAnimId);
        drawWaveform();

        // Timer
        recordInterval = setInterval(() => {
            recordSeconds++;
            timerDisplay.textContent = formatTime(recordSeconds);
            if (recordSeconds >= MAX_RECORD_SECONDS) {
                stopRecording();
            }
        }, 1000);

    } catch (err) {
        console.error('Microphone error:', err);
        let msg = 'Microphone access denied. Please check browser permissions.';
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
            msg = 'Permission denied. Click the lock/settings icon beside the URL in your browser address bar and set Microphone to "Allow".';
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
            msg = 'No microphone detected. Please plug in a microphone or use "Upload Audio File".';
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
            msg = 'Microphone is currently in use by another app. Please close other audio apps and try again.';
        }
        statusText.textContent = msg;
        statusIndicator.className = 'status-indicator';
        alert(`Microphone Error: ${msg}`);
    }
}

function stopRecording() {
    isRecording = false;

    if (recordInterval) {
        clearInterval(recordInterval);
        recordInterval = null;
    }
    cancelAnimationFrame(waveformAnimId);

    // Collect all samples into one Float32Array
    let totalLength = 0;
    for (const chunk of recordedSamples) {
        totalLength += chunk.length;
    }

    const mergedSamples = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of recordedSamples) {
        mergedSamples.set(chunk, offset);
        offset += chunk.length;
    }

    const sampleRate = audioContext ? audioContext.sampleRate : 44100;

    // Disconnect and clean audio context
    if (scriptProcessor) {
        scriptProcessor.disconnect();
        scriptProcessor = null;
    }
    if (sourceNode) {
        sourceNode.disconnect();
        sourceNode = null;
    }
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }

    updateUI('analyzing');

    // Send uncompressed native audio (backend uses librosa sinc anti-aliased 16kHz resampling)
    const wavBlob = encodeWAV(mergedSamples, sampleRate);

    // Send WAV audio to backend
    sendAudioForPrediction(wavBlob);
}

// ==========================================
// API Communication
// ==========================================
async function sendAudioForPrediction(blob) {
    showAnalyzingSteps();

    const formData = new FormData();
    const fileName = (blob instanceof File && blob.name) ? blob.name : 'recording.wav';
    formData.append('audio', blob, fileName);

    try {
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (!response.ok || result.error) {
            throw new Error(result.error || `Server error: ${response.status}`);
        }

        await completeAnalyzingSteps();
        displayResults(result);

    } catch (err) {
        console.error('Prediction error:', err);
        statusText.textContent = `Error: ${err.message}`;
        statusIndicator.className = 'status-indicator';
        analyzingState.classList.add('hidden');
        alert(`Analysis failed: ${err.message}`);
        updateUI('ready');
    }
}

// ==========================================
// Analyzing Steps Animation
// ==========================================
function showAnalyzingSteps() {
    analyzingState.classList.remove('hidden');
    const steps = [
        document.getElementById('step1'),
        document.getElementById('step2'),
        document.getElementById('step3'),
    ];

    steps.forEach(s => {
        if (s) s.className = 'step-item';
    });

    if (steps[0]) setTimeout(() => steps[0].classList.add('active'), 150);
    if (steps[1]) setTimeout(() => {
        steps[0].classList.remove('active');
        steps[0].classList.add('done');
        steps[1].classList.add('active');
    }, 900);
    if (steps[2]) setTimeout(() => {
        steps[1].classList.remove('active');
        steps[1].classList.add('done');
        steps[2].classList.add('active');
    }, 1800);
}

function completeAnalyzingSteps() {
    return new Promise(resolve => {
        const step3 = document.getElementById('step3');
        if (step3) {
            step3.classList.remove('active');
            step3.classList.add('done');
        }
        setTimeout(resolve, 400);
    });
}

// ==========================================
// Display Results
// ==========================================
function displayResults(result) {
    updateUI('done');
    analyzingState.classList.add('hidden');
    resultsCard.classList.remove('hidden');

    const isReal = result.label === 'Real';
    const confidence = result.confidence || 0;

    // Dynamically update SVG gauge color
    const svgNS = 'http://www.w3.org/2000/svg';
    let existingGrad = document.getElementById('gaugeGradient');
    if (existingGrad) existingGrad.remove();

    const gaugeSvg = gaugeFill.closest('svg');
    const defs = document.createElementNS(svgNS, 'defs');
    const gradient = document.createElementNS(svgNS, 'linearGradient');
    gradient.setAttribute('id', 'gaugeGradient');
    gradient.setAttribute('x1', '0%');
    gradient.setAttribute('y1', '0%');
    gradient.setAttribute('x2', '100%');
    gradient.setAttribute('y2', '100%');

    const stop1 = document.createElementNS(svgNS, 'stop');
    const stop2 = document.createElementNS(svgNS, 'stop');
    stop1.setAttribute('offset', '0%');
    stop2.setAttribute('offset', '100%');

    if (isReal) {
        stop1.setAttribute('stop-color', '#10b981');
        stop2.setAttribute('stop-color', '#34d399');
    } else {
        stop1.setAttribute('stop-color', '#ef4444');
        stop2.setAttribute('stop-color', '#f97316');
    }

    gradient.appendChild(stop1);
    gradient.appendChild(stop2);
    defs.appendChild(gradient);
    gaugeSvg.insertBefore(defs, gaugeSvg.firstChild);

    // Gauge circle animation
    const circumference = 2 * Math.PI * 85;
    const offset = circumference - (confidence / 100) * circumference;
    gaugeFill.setAttribute('stroke', 'url(#gaugeGradient)');

    setTimeout(() => {
        gaugeFill.style.strokeDashoffset = offset;
    }, 100);

    // Number counter animation
    animateCounter(gaugeValue, 0, Math.round(confidence), 1400);

    // Verdict label & description
    verdictLabel.textContent = isReal ? '✅ REAL VOICE' : '⚠️ FAKE VOICE';
    verdictLabel.className = `verdict-label ${isReal ? 'real' : 'fake'}`;
    verdictDesc.textContent = isReal
        ? 'This voice appears authentic. Natural vocal dynamics and human harmonic resonance detected.'
        : 'This voice exhibits synthetic characteristics. Spectral artifacts and artificial textures detected.';

    // Probability bars
    const realProb = result.real_probability !== undefined ? result.real_probability : (isReal ? confidence : 100 - confidence);
    const fakeProb = result.fake_probability !== undefined ? result.fake_probability : (isReal ? 100 - confidence : confidence);

    setTimeout(() => {
        realBar.style.width = `${realProb}%`;
        fakeBar.style.width = `${fakeProb}%`;
    }, 300);
    realValue.textContent = `${realProb.toFixed(1)}%`;
    fakeValue.textContent = `${fakeProb.toFixed(1)}%`;

    // Render Deep Learning Forward Propagation Stages
    const forwardStagesGrid = document.getElementById('forwardStagesGrid');
    const dnnArchText = document.getElementById('dnnArchText');
    if (result.deep_learning_model && dnnArchText) {
        dnnArchText.textContent = `${result.deep_learning_model.architecture || 'Deep Neural Network'} • ${result.deep_learning_model.hidden_layers || 7} Hidden Layers • ${result.deep_learning_model.forward_propagation_stages || 8} Forward Stages`;
    }

    if (forwardStagesGrid) {
        forwardStagesGrid.innerHTML = '';
        if (result.forward_propagation_stages && result.forward_propagation_stages.length > 0) {
            result.forward_propagation_stages.forEach((stage, i) => {
                const card = document.createElement('div');
                card.className = 'stage-card';
                card.style.animation = `slideUp 0.4s ease forwards ${i * 0.06}s`;
                card.innerHTML = `
                    <div class="stage-card-header">
                        <span class="stage-badge">Stage ${stage.stage}</span>
                        <span class="stage-neurons">${stage.neurons} neurons</span>
                    </div>
                    <div class="stage-name">${stage.name}</div>
                    <div class="stage-meta">
                        <span class="stage-flow">${stage.flow}</span>
                        <span class="stage-act">${stage.activation}</span>
                    </div>
                `;
                forwardStagesGrid.appendChild(card);
            });
        }
    }

    // Feature breakdown chips
    featureGrid.innerHTML = '';
    if (result.features_analyzed && result.features_analyzed.length > 0) {
        result.features_analyzed.forEach((feat, i) => {
            const chip = document.createElement('div');
            chip.className = 'feature-chip';
            chip.style.animation = `slideUp 0.4s ease forwards ${i * 0.08}s`;
            chip.innerHTML = `
                <span class="feature-chip-name" title="${feat.description || ''}">${feat.name}</span>
                <span class="feature-chip-value">${feat.importance}%</span>
            `;
            featureGrid.appendChild(chip);
        });
    }

    // Smooth scroll to results
    setTimeout(() => {
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 200);
}

// ==========================================
// Counter Animation
// ==========================================
function animateCounter(element, start, end, duration) {
    const range = end - start;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(start + range * eased);
        element.textContent = current;
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    requestAnimationFrame(update);
}

// ==========================================
// UI State Management
// ==========================================
function updateUI(state) {
    switch (state) {
        case 'recording':
            micButton.classList.add('recording');
            micIcon.classList.add('hidden');
            stopIcon.classList.remove('hidden');
            micRipple.classList.add('active');
            statusIndicator.className = 'status-indicator recording';
            statusText.textContent = 'Recording... Click button again when done';
            resultsCard.classList.add('hidden');
            break;

        case 'analyzing':
            micButton.classList.remove('recording');
            micIcon.classList.remove('hidden');
            stopIcon.classList.add('hidden');
            micRipple.classList.remove('active');
            statusIndicator.className = 'status-indicator analyzing';
            statusText.textContent = 'Analyzing voice features...';
            break;

        case 'done':
            statusIndicator.className = 'status-indicator done';
            statusText.textContent = 'Analysis complete!';
            break;

        case 'ready':
        default:
            micButton.classList.remove('recording');
            micIcon.classList.remove('hidden');
            stopIcon.classList.add('hidden');
            micRipple.classList.remove('active');
            statusIndicator.className = 'status-indicator';
            statusText.textContent = 'Ready to record';
            timerDisplay.textContent = '00:00';
            analyzingState.classList.add('hidden');
            resultsCard.classList.add('hidden');
            drawIdleWaveform();
            break;
    }
}

// ==========================================
// Reset Button
// ==========================================
resetBtn.addEventListener('click', () => {
    gaugeFill.style.strokeDashoffset = 534;
    gaugeValue.textContent = '0';
    realBar.style.width = '0%';
    fakeBar.style.width = '0%';
    updateUI('ready');
});

// ==========================================
// Mode Tabs Switcher (Record vs Upload)
// ==========================================
tabRecord.addEventListener('click', () => {
    tabRecord.classList.add('active');
    tabUpload.classList.remove('active');
    recordModeContainer.classList.remove('hidden');
    uploadModeContainer.classList.add('hidden');
    statusText.textContent = 'Ready to record';
});

tabUpload.addEventListener('click', () => {
    tabUpload.classList.add('active');
    tabRecord.classList.remove('active');
    recordModeContainer.classList.add('hidden');
    uploadModeContainer.classList.remove('hidden');
    statusText.textContent = 'Ready to upload audio file';
    if (isRecording) stopRecording();
});

// File input & Drag-and-drop
audioDropzone.addEventListener('click', () => audioFileInput.click());

audioFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
        handleFileSelection(e.target.files[0]);
    }
});

['dragenter', 'dragover'].forEach(eventName => {
    audioDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        audioDropzone.classList.add('dragover');
    });
});

['dragleave', 'drop'].forEach(eventName => {
    audioDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        audioDropzone.classList.remove('dragover');
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });
});

function handleFileSelection(file) {
    selectedAudioFile = file;
    selectedFileName.textContent = file.name;
    selectedFileInfo.classList.remove('hidden');
    statusText.textContent = `File selected: ${file.name}`;
}

btnAnalyzeFile.addEventListener('click', () => {
    if (!selectedAudioFile) return;
    sendAudioForPrediction(selectedAudioFile);
});

// ==========================================
// FAQ Accordion
// ==========================================
document.querySelectorAll('.faq-question').forEach(btn => {
    btn.addEventListener('click', () => {
        const item = btn.closest('.faq-item');
        const isOpen = item.classList.contains('open');
        document.querySelectorAll('.faq-item').forEach(faq => faq.classList.remove('open'));
        if (!isOpen) {
            item.classList.add('open');
        }
    });
});

// ==========================================
// Smooth Scroll for Nav Links
// ==========================================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
        e.preventDefault();
        const target = document.querySelector(anchor.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});

// ==========================================
// Utilities
// ==========================================
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
    const secs = (seconds % 60).toString().padStart(2, '0');
    return `${mins}:${secs}`;
}

// ==========================================
// Navbar Scroll Effect
// ==========================================
window.addEventListener('scroll', () => {
    const navbar = document.getElementById('navbar');
    if (window.scrollY > 50) {
        navbar.style.background = 'rgba(10, 14, 26, 0.95)';
        navbar.style.borderBottomColor = 'rgba(255, 255, 255, 0.08)';
    } else {
        navbar.style.background = 'rgba(10, 14, 26, 0.85)';
        navbar.style.borderBottomColor = 'rgba(255, 255, 255, 0.06)';
    }
});
