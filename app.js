/* Lensless — three browser-side ML demos.
   Uses transformers.js from a CDN. ES modules. */

import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2";

// Force WASM by default; the runtime will pick WebGPU when available and supported.
env.allowLocalModels = false;
env.useBrowserCache = true;

const $  = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => Array.from(ctx.querySelectorAll(sel));

const live = $("#live");
const announce = (m) => { if (live) live.textContent = m; };

/* ============================================================
   TABS — APG pattern with arrow keys
   ============================================================ */
function wireTabs() {
  const tablist = $('[role="tablist"]');
  const tabs = $$('[role="tab"]', tablist);
  const panels = tabs.map((t) => $("#" + t.getAttribute("aria-controls")));

  const select = (idx, { focus = true } = {}) => {
    tabs.forEach((t, i) => {
      const on = i === idx;
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.setAttribute("tabindex", on ? "0" : "-1");
      t.classList.toggle("is-on", on);
      panels[i].hidden = !on;
    });
    if (focus) tabs[idx].focus();
  };

  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => select(i, { focus: false }));
    tab.addEventListener("keydown", (e) => {
      let target = null;
      switch (e.key) {
        case "ArrowRight": target = (i + 1) % tabs.length; break;
        case "ArrowLeft":  target = (i - 1 + tabs.length) % tabs.length; break;
        case "Home":       target = 0; break;
        case "End":        target = tabs.length - 1; break;
      }
      if (target !== null) { e.preventDefault(); select(target); }
    });
  });
}

/* ============================================================
   Cached pipelines — load on first use
   ============================================================ */
const pipelines = {};

async function getPipeline(task, model, onProgress) {
  const key = task + "::" + model;
  if (pipelines[key]) return pipelines[key];
  pipelines[key] = await pipeline(task, model, {
    progress_callback: (p) => onProgress && onProgress(p),
  });
  return pipelines[key];
}

function setStatus(el, text, cls) {
  el.textContent = text;
  el.className = "status" + (cls ? " " + cls : "");
}

function setBusy(node, busy) {
  if (!node) return;
  node.setAttribute("aria-busy", busy ? "true" : "false");
}

function setBtnRunning(btn, running, idleLabel) {
  if (!btn) return;
  btn.setAttribute("aria-disabled", running ? "true" : "false");
  btn.textContent = running ? "Running…" : idleLabel;
}

/* Helper: turn a URL into an HTMLImageElement once it has loaded */
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Image failed to load."));
    img.src = src;
  });
}

function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(new Error("Could not read file."));
    r.readAsDataURL(file);
  });
}

/* ============================================================
   CLASSIFY — image classification with MobileNet
   ============================================================ */
function wireClassify() {
  const fileInput = $("#classify-file");
  const imgEl = $("#classify-img");
  const placeholder = $("#classify-placeholder");
  const statusEl = $("#classify-status");
  const predsEl = $("#classify-preds");
  const resultH = $("#classify-result-h");
  const timingEl = $("#classify-timing");
  const result = $("#classify-result");

  // The Xenova/mobilenet_v2_1.0_224 catalogue entry was removed from the
  // Hugging Face Hub. Switched to Xenova/vit-base-patch16-224 which is still
  // reachable and ships with a quantised ONNX model file usable from
  // transformers.js. Slightly larger download (~85 MB) but still tractable.
  const MODEL = "Xenova/vit-base-patch16-224";

  const showImage = (src, label = "Uploaded image") => {
    imgEl.src = src;
    imgEl.hidden = false;
    imgEl.alt = label;
    placeholder.hidden = true;
  };

  const renderProgress = (p) => {
    if (p.status === "progress" && p.total) {
      const pct = Math.round((p.loaded / p.total) * 100);
      const mbDone = (p.loaded / 1024 / 1024).toFixed(1);
      const mbTotal = (p.total / 1024 / 1024).toFixed(1);
      setStatus(statusEl, `Downloading model: ${pct}% (${mbDone} of ${mbTotal} MB).`, "is-progress");
    } else if (p.status === "ready") {
      setStatus(statusEl, "Model ready. Running inference.", "is-progress");
    }
  };

  const run = async (src, label) => {
    showImage(src, label);
    setStatus(statusEl, "Loading model. First use only.", "is-progress");
    setBusy(result, true);
    predsEl.innerHTML = "";
    resultH.hidden = true;
    timingEl.textContent = "";

    try {
      const classifier = await getPipeline("image-classification", MODEL, renderProgress);
      setStatus(statusEl, "Running inference.", "is-progress");
      const t0 = performance.now();
      const out = await classifier(src, { topk: 5 });
      const ms = Math.round(performance.now() - t0);

      predsEl.innerHTML = out.map((o, i) => `
        <li>
          <span class="pred-rank">${i + 1}</span>
          <span class="pred-label">${o.label}</span>
          <span class="pred-pct">${(o.score * 100).toFixed(1)}%</span>
        </li>
      `).join("");
      resultH.hidden = false;
      const top = out[0];
      imgEl.alt = `Uploaded image. Top label: ${top.label}, ${(top.score * 100).toFixed(0)} percent confidence.`;
      setStatus(statusEl, `Done. Top label: ${top.label}.`, "");
      timingEl.textContent = `Inference ${ms} ms. Running locally in your browser.`;
      announce(`Top label: ${top.label}, ${(top.score * 100).toFixed(0)} percent confidence.`);
    } catch (err) {
      console.error(err);
      setStatus(statusEl, `Could not classify the image. ${err.message || err}`, "is-error");
    } finally {
      setBusy(result, false);
    }
  };

  fileInput.addEventListener("change", async (e) => {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    if (f.size > 8 * 1024 * 1024) {
      setStatus(statusEl, "Image is larger than 8 MB. Try a smaller file.", "is-error");
      return;
    }
    const src = await fileToDataURL(f);
    run(src, `Uploaded image: ${f.name}`);
  });

  $$('[data-sample="classify"]').forEach((btn) => {
    btn.addEventListener("click", () => run(btn.dataset.img, `Sample image: ${btn.textContent.trim()}`));
  });
}

/* ============================================================
   DETECT — object detection with DETR
   ============================================================ */
function wireDetect() {
  const fileInput = $("#detect-file");
  const imgEl = $("#detect-img");
  const placeholder = $("#detect-placeholder");
  const overlay = $("#detect-overlay");
  const statusEl = $("#detect-status");
  const listEl = $("#detect-list");
  const resultH = $("#detect-result-h");
  const timingEl = $("#detect-timing");
  const result = $("#detect-result");

  const MODEL = "Xenova/detr-resnet-50";

  const renderProgress = (p) => {
    if (p.status === "progress" && p.total) {
      const pct = Math.round((p.loaded / p.total) * 100);
      const mbDone = (p.loaded / 1024 / 1024).toFixed(1);
      const mbTotal = (p.total / 1024 / 1024).toFixed(1);
      setStatus(statusEl, `Downloading model: ${pct}% (${mbDone} of ${mbTotal} MB). This is a one-time download.`, "is-progress");
    } else if (p.status === "ready") {
      setStatus(statusEl, "Model ready. Running inference.", "is-progress");
    }
  };

  const drawBoxes = (boxes, naturalW, naturalH) => {
    overlay.innerHTML = "";
    overlay.setAttribute("viewBox", `0 0 ${naturalW} ${naturalH}`);
    overlay.setAttribute("preserveAspectRatio", "xMidYMid meet");
    boxes.forEach((b) => {
      const x = b.box.xmin, y = b.box.ymin;
      const w = b.box.xmax - b.box.xmin, h = b.box.ymax - b.box.ymin;
      const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      r.setAttribute("class", "det-box");
      r.setAttribute("x", x); r.setAttribute("y", y);
      r.setAttribute("width", w); r.setAttribute("height", h);
      overlay.appendChild(r);

      const labelText = `${b.label} ${(b.score * 100).toFixed(0)}%`;
      const padX = Math.max(4, naturalW * 0.005);
      const padY = Math.max(4, naturalH * 0.008);
      const fontSize = Math.max(14, Math.round(naturalH * 0.022));
      const tw = labelText.length * fontSize * 0.55 + padX * 2;
      const th = fontSize + padY * 1.5;

      const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      bg.setAttribute("class", "det-label-bg");
      bg.setAttribute("x", x);
      bg.setAttribute("y", Math.max(0, y - th));
      bg.setAttribute("width", Math.min(tw, w));
      bg.setAttribute("height", th);
      overlay.appendChild(bg);

      const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("class", "det-label");
      t.setAttribute("x", x + padX);
      t.setAttribute("y", Math.max(fontSize, y - padY));
      t.setAttribute("font-size", fontSize);
      t.textContent = labelText;
      overlay.appendChild(t);
    });
  };

  const positionFor = (b, w, h) => {
    const cx = (b.box.xmin + b.box.xmax) / 2;
    const cy = (b.box.ymin + b.box.ymax) / 2;
    const horizontal = cx < w * 0.33 ? "left" : cx > w * 0.66 ? "right" : "centre";
    const vertical = cy < h * 0.33 ? "upper" : cy > h * 0.66 ? "lower" : "middle";
    return `${vertical} ${horizontal}`;
  };

  const run = async (src, label) => {
    imgEl.src = src;
    imgEl.alt = label;
    imgEl.hidden = false;
    placeholder.hidden = true;
    overlay.innerHTML = "";
    listEl.innerHTML = "";
    resultH.hidden = true;
    timingEl.textContent = "";
    setStatus(statusEl, "Loading model. First use only.", "is-progress");
    setBusy(result, true);

    try {
      const detector = await getPipeline("object-detection", MODEL, renderProgress);
      // Wait for the image to settle so we can read natural dimensions
      await new Promise((resolve) => {
        if (imgEl.complete && imgEl.naturalWidth) resolve();
        else imgEl.onload = () => resolve();
      });

      setStatus(statusEl, "Running inference.", "is-progress");
      const t0 = performance.now();
      const out = await detector(src, { threshold: 0.6, percentage: false });
      const ms = Math.round(performance.now() - t0);

      const W = imgEl.naturalWidth, H = imgEl.naturalHeight;
      drawBoxes(out, W, H);

      if (out.length === 0) {
        setStatus(statusEl, "No objects detected above 60 percent confidence. Try a different image.", "");
      } else {
        listEl.innerHTML = out.map((b) => `
          <li>
            <span class="det-name">${b.label}</span>
            <span class="det-conf">${(b.score * 100).toFixed(0)}% · ${positionFor(b, W, H)}</span>
          </li>
        `).join("");
        resultH.hidden = false;
        const summary = out.slice(0, 3).map((b) => `${b.label} ${(b.score * 100).toFixed(0)} percent`).join(", ");
        imgEl.alt = `Uploaded image. Detected: ${summary}.`;
        setStatus(statusEl, `Done. Detected ${out.length} object${out.length === 1 ? "" : "s"}.`, "");
        announce(`Detected ${out.length} objects. Top: ${out[0].label}.`);
      }
      timingEl.textContent = `Inference ${ms} ms. Running locally in your browser.`;
    } catch (err) {
      console.error(err);
      setStatus(statusEl, `Could not detect objects. ${err.message || err}`, "is-error");
    } finally {
      setBusy(result, false);
    }
  };

  fileInput.addEventListener("change", async (e) => {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    if (f.size > 8 * 1024 * 1024) {
      setStatus(statusEl, "Image is larger than 8 MB. Try a smaller file.", "is-error");
      return;
    }
    const src = await fileToDataURL(f);
    run(src, `Uploaded image: ${f.name}`);
  });

  $$('[data-sample="detect"]').forEach((btn) => {
    btn.addEventListener("click", () => run(btn.dataset.img, `Sample image: ${btn.textContent.trim()}`));
  });
}

/* ============================================================
   SENTIMENT — DistilBERT
   ============================================================ */
function wireSentiment() {
  const input = $("#sent-input");
  const runBtn = $("#sent-run");
  const clearBtn = $("#sent-clear");
  const statusEl = $("#sent-status");
  const out = $("#sent-out");
  const verdict = $("#sent-verdict");
  const barPos = $("#sent-bar-pos");
  const barNeg = $("#sent-bar-neg");
  const barPosVal = $("#sent-bar-pos-val");
  const barNegVal = $("#sent-bar-neg-val");
  const timingEl = $("#sent-timing");
  const result = $("#sent-result");

  const MODEL = "Xenova/distilbert-base-uncased-finetuned-sst-2-english";

  const renderProgress = (p) => {
    if (p.status === "progress" && p.total) {
      const pct = Math.round((p.loaded / p.total) * 100);
      const mbDone = (p.loaded / 1024 / 1024).toFixed(1);
      const mbTotal = (p.total / 1024 / 1024).toFixed(1);
      setStatus(statusEl, `Downloading model: ${pct}% (${mbDone} of ${mbTotal} MB).`, "is-progress");
    } else if (p.status === "ready") {
      setStatus(statusEl, "Model ready. Running inference.", "is-progress");
    }
  };

  const run = async () => {
    const text = input.value.trim();
    if (!text) {
      setStatus(statusEl, "Enter some text first.", "is-error");
      return;
    }
    setBtnRunning(runBtn, true, "Analyse sentiment");
    setStatus(statusEl, "Loading model. First use only.", "is-progress");
    setBusy(result, true);
    out.hidden = true;

    try {
      const classifier = await getPipeline("sentiment-analysis", MODEL, renderProgress);
      const t0 = performance.now();
      const r = await classifier(text);
      const ms = Math.round(performance.now() - t0);
      const top = r[0];
      const isPos = top.label.toUpperCase() === "POSITIVE";
      const posPct = isPos ? top.score : 1 - top.score;
      const negPct = 1 - posPct;

      verdict.textContent = `${isPos ? "Positive" : "Negative"} sentiment, ${(top.score * 100).toFixed(0)} percent confidence.`;
      verdict.className = "sent-verdict " + (isPos ? "positive" : "negative");
      barPos.style.width = (posPct * 100).toFixed(1) + "%";
      barNeg.style.width = (negPct * 100).toFixed(1) + "%";
      barPosVal.textContent = (posPct * 100).toFixed(0) + "%";
      barNegVal.textContent = (negPct * 100).toFixed(0) + "%";
      timingEl.textContent = `Inference ${ms} ms. Running locally in your browser.`;
      out.hidden = false;
      setStatus(statusEl, "Done.", "");
      announce(verdict.textContent);
    } catch (err) {
      console.error(err);
      setStatus(statusEl, `Could not analyse the text. ${err.message || err}`, "is-error");
    } finally {
      setBtnRunning(runBtn, false, "Analyse sentiment");
      setBusy(result, false);
    }
  };

  runBtn.addEventListener("click", run);
  clearBtn.addEventListener("click", () => {
    input.value = "";
    out.hidden = true;
    setStatus(statusEl, "Cleared. Ready for new text.", "");
    input.focus();
    announce("Cleared.");
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      run();
    }
  });
}

/* ---------- boot ---------- */
function init() {
  wireTabs();
  wireClassify();
  wireDetect();
  wireSentiment();
  announce("Lensless ready.");
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
