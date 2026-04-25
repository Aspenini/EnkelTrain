import "./styles.css";
import {
  createTrainingData7z,
  createTrainingDataZip,
  documentsToCorpus,
  loadTrainingDocuments,
  type TrainingDocument
} from "./archive";
import { createSafetensorsBlob } from "./safetensors";
import { BpeTokenizer, trainBpeTokenizer } from "./tokenizer";
import { initializeBestBackend, TinyGpt, type TinyGptConfig } from "./tiny-gpt";

type AppState = {
  backend: string;
  documents: TrainingDocument[];
  corpus: string;
  tokenizer: BpeTokenizer | null;
  model: TinyGpt | null;
  trained: boolean;
  busy: boolean;
  lastTrainingSummary: Record<string, string>;
};

const state: AppState = {
  backend: "checking",
  documents: [],
  corpus: "",
  tokenizer: null,
  model: null,
  trained: false,
  busy: false,
  lastTrainingSummary: {}
};

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) {
  throw new Error("Missing #app root");
}

app.innerHTML = `
  <main class="shell">
    <section class="topbar">
      <div>
        <h1>EnkelTrain</h1>
        <p>Train a tiny GPT-style language model in your browser, then run it locally.</p>
      </div>
      <div class="topbar-actions">
        <a
          class="repo-button"
          href="https://github.com/Aspenini/EnkelTrain"
          target="_blank"
          rel="noreferrer"
        >
          View on GitHub
        </a>
        <div class="status-block">
          <span class="status-dot" id="backend-dot"></span>
          <span id="backend-label">Backend: checking</span>
        </div>
      </div>
    </section>

    <section class="workspace">
      <aside class="panel controls">
        <div class="section-head">
          <h2>Training Data</h2>
          <span id="file-count">0 files</span>
        </div>
        <label class="drop-zone" for="file-input" id="drop-zone">
          <input id="file-input" type="file" multiple accept=".txt,.md,.markdown,.csv,.json,.jsonl,.html,.xml,.js,.ts,.tsx,.jsx,.py,.rs,.go,.java,.c,.cpp,.h,.hpp,.css,.scss,.sql,.yaml,.yml,.toml,.ini,.log,.zip,.7z,application/zip,application/x-7z-compressed" />
          <strong>Select text or archives</strong>
          <span>Plain text, Markdown, code, CSV, JSON, logs, ZIP, and 7z work best.</span>
        </label>
        <button id="sample-button" class="button secondary compact" type="button">Use sample data</button>
        <div class="split-buttons">
          <button id="export-zip-button" class="button secondary compact" type="button" disabled>Export data .zip</button>
          <button id="export-7z-button" class="button secondary compact" type="button" disabled>Export data .7z</button>
        </div>
        <div id="file-list" class="file-list empty">No files loaded.</div>

        <div class="section-head">
          <h2>Tokenizer</h2>
        </div>
        <label class="field">
          <span>Vocabulary size</span>
          <input id="vocab-size" type="number" min="260" max="2048" step="16" value="512" />
        </label>
        <button id="tokenizer-button" class="button secondary" type="button">Train tokenizer</button>
        <p id="tokenizer-stats" class="small-muted">Tokenizer not trained.</p>

        <div class="section-head">
          <h2>Model</h2>
        </div>
        <div class="grid-2">
          <label class="field">
            <span>Context</span>
            <input id="context-size" type="number" min="8" max="96" step="8" value="32" />
          </label>
          <label class="field">
            <span>Width</span>
            <input id="d-model" type="number" min="24" max="160" step="8" value="64" />
          </label>
          <label class="field">
            <span>Layers</span>
            <input id="layers" type="number" min="1" max="3" step="1" value="1" />
          </label>
          <label class="field">
            <span>Epochs</span>
            <input id="epochs" type="number" min="1" max="20" step="1" value="4" />
          </label>
          <label class="field">
            <span>Batch</span>
            <input id="batch-size" type="number" min="4" max="64" step="4" value="16" />
          </label>
          <label class="field">
            <span>Steps / epoch</span>
            <input id="steps" type="number" min="10" max="1000" step="10" value="80" />
          </label>
        </div>
        <label class="field">
          <span>Learning rate</span>
          <input id="learning-rate" type="number" min="0.0001" max="0.01" step="0.0001" value="0.0015" />
        </label>
        <button id="train-button" class="button primary" type="button">Train model</button>
        <div class="progress">
          <div id="progress-bar"></div>
        </div>
        <p id="training-log" class="small-muted">Waiting for data.</p>
        <button id="download-button" class="button secondary" type="button" disabled>Download .safetensors</button>
      </aside>

      <section class="panel chat">
        <div class="section-head">
          <h2>Local Chat</h2>
          <span id="chat-state">Train a model first</span>
        </div>
        <div id="chat-log" class="chat-log">
          <div class="message system">Load text files, train the tokenizer and model, then prompt your local model here.</div>
        </div>
        <div class="prompt-row">
          <textarea id="prompt-input" placeholder="Ask your trained model something..." rows="3"></textarea>
          <button id="generate-button" class="button primary" type="button" disabled>Run</button>
        </div>
        <div class="generation-controls">
          <label class="field">
            <span>New tokens</span>
            <input id="max-new-tokens" type="number" min="8" max="256" step="8" value="96" />
          </label>
          <label class="field">
            <span>Temperature</span>
            <input id="temperature" type="number" min="0.1" max="2" step="0.1" value="0.8" />
          </label>
          <label class="field">
            <span>Top K</span>
            <input id="top-k" type="number" min="1" max="80" step="1" value="20" />
          </label>
        </div>
      </section>
    </section>
  </main>
`;

const byId = <T extends HTMLElement>(id: string) => {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing #${id}`);
  }
  return element as T;
};

const backendDot = byId<HTMLSpanElement>("backend-dot");
const backendLabel = byId<HTMLSpanElement>("backend-label");
const fileInput = byId<HTMLInputElement>("file-input");
const dropZone = byId<HTMLLabelElement>("drop-zone");
const sampleButton = byId<HTMLButtonElement>("sample-button");
const exportZipButton = byId<HTMLButtonElement>("export-zip-button");
const export7zButton = byId<HTMLButtonElement>("export-7z-button");
const fileCount = byId<HTMLSpanElement>("file-count");
const fileList = byId<HTMLDivElement>("file-list");
const tokenizerButton = byId<HTMLButtonElement>("tokenizer-button");
const tokenizerStats = byId<HTMLParagraphElement>("tokenizer-stats");
const trainButton = byId<HTMLButtonElement>("train-button");
const trainingLog = byId<HTMLParagraphElement>("training-log");
const progressBar = byId<HTMLDivElement>("progress-bar");
const downloadButton = byId<HTMLButtonElement>("download-button");
const chatState = byId<HTMLSpanElement>("chat-state");
const chatLog = byId<HTMLDivElement>("chat-log");
const promptInput = byId<HTMLTextAreaElement>("prompt-input");
const generateButton = byId<HTMLButtonElement>("generate-button");

function numberInput(id: string) {
  return byId<HTMLInputElement>(id).valueAsNumber;
}

function setBusy(busy: boolean) {
  state.busy = busy;
  sampleButton.disabled = busy;
  exportZipButton.disabled = busy || state.documents.length === 0;
  export7zButton.disabled = busy || state.documents.length === 0;
  tokenizerButton.disabled = busy || state.corpus.length === 0;
  trainButton.disabled = busy || state.corpus.length === 0;
  generateButton.disabled = busy || !state.trained;
  downloadButton.disabled = busy || !state.trained;
}

function renderBackend() {
  backendDot.className = `status-dot ${state.backend}`;
  backendLabel.textContent = `Backend: ${state.backend}`;
}

function renderFiles() {
  fileCount.textContent = `${state.documents.length} file${state.documents.length === 1 ? "" : "s"}`;
  if (state.documents.length === 0) {
    fileList.className = "file-list empty";
    fileList.textContent = "No files loaded.";
    return;
  }

  fileList.className = "file-list";
  fileList.innerHTML = state.documents
    .map(
      (document) =>
        `<div><span title="${escapeHtml(document.path)}">${escapeHtml(document.path)}</span><small>${formatBytes(document.size)}</small></div>`
    )
    .join("");
}

function appendMessage(kind: "user" | "assistant" | "system", text: string) {
  const message = document.createElement("div");
  message.className = `message ${kind}`;
  message.textContent = text;
  chatLog.append(message);
  chatLog.scrollTop = chatLog.scrollHeight;
  return message;
}

function escapeHtml(value: string) {
  return value.replace(/[&<>"']/g, (char) => {
    const entities: Record<string, string> = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;"
    };
    return entities[char];
  });
}

function formatBytes(size: number) {
  if (size < 1024) {
    return `${size} B`;
  }
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  }
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

async function readFiles(files: FileList | File[]) {
  state.documents = [];
  state.corpus = "";
  state.tokenizer = null;
  state.model = null;
  state.trained = false;
  tokenizerStats.textContent = "Tokenizer not trained.";
  trainingLog.textContent = files.length ? "Reading files and archives..." : "No files selected.";
  progressBar.style.width = "0%";
  renderFiles();
  setBusy(true);

  try {
    const { documents, rejected } = await loadTrainingDocuments(files);
    state.documents = documents;
    state.corpus = await documentsToCorpus(documents);
    renderFiles();

    const rejectedText = rejected.length ? ` Ignored ${rejected.length} unsupported item${rejected.length === 1 ? "" : "s"}.` : "";
    trainingLog.textContent = state.corpus.length
      ? `Loaded ${documents.length} text file${documents.length === 1 ? "" : "s"} with ${state.corpus.length.toLocaleString()} characters.${rejectedText}`
      : `No readable text found.${rejectedText}`;
  } catch (error) {
    trainingLog.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    setBusy(false);
  }
}

function useSampleData() {
  const sampleText = [
    "EnkelTrain runs entirely inside the browser.",
    "A tiny language model learns short patterns from local text.",
    "The tokenizer turns text into compact pieces before training.",
    "WebGPU can accelerate tensor math when the browser supports it.",
    "After training, the model writes continuations from a prompt.",
    "Local training keeps the selected data on this computer."
  ].join("\n");
  const file = new File([sampleText], "sample-training-data.txt", { type: "text/plain" });
  state.documents = [{ name: file.name, path: file.name, size: file.size, file }];
  state.corpus = sampleText;
  state.tokenizer = null;
  state.model = null;
  state.trained = false;
  tokenizerStats.textContent = "Tokenizer not trained.";
  trainingLog.textContent = `Loaded sample corpus with ${state.corpus.length.toLocaleString()} characters.`;
  progressBar.style.width = "0%";
  chatState.textContent = "Train a model first";
  renderFiles();
  setBusy(false);
}

function trainTokenizer() {
  if (!state.corpus) {
    tokenizerStats.textContent = "Select text files before training a tokenizer.";
    return;
  }

  setBusy(true);
  const targetVocabSize = numberInput("vocab-size");
  const { tokenizer, stats } = trainBpeTokenizer(state.corpus, targetVocabSize);
  state.tokenizer = tokenizer;
  state.trained = false;
  state.model = null;
  tokenizerStats.textContent = `Byte-level GPT-style BPE: ${stats.vocabSize} tokens, ${stats.mergeCount} merges, ${stats.pretokenCount.toLocaleString()} pretokens.`;
  trainingLog.textContent = "Tokenizer trained. Model is ready to train.";
  chatState.textContent = "Train a model first";
  progressBar.style.width = "0%";
  setBusy(false);
}

async function trainModel() {
  if (!state.corpus) {
    trainingLog.textContent = "Select text files before training.";
    return;
  }

  if (!state.tokenizer) {
    trainTokenizer();
  }

  if (!state.tokenizer) {
    return;
  }

  setBusy(true);
  state.trained = false;
  chatState.textContent = "Training";
  progressBar.style.width = "0%";

  try {
    const contextSize = numberInput("context-size");
    const config: TinyGptConfig = {
      vocabSize: state.tokenizer.vocab.length,
      contextSize,
      dModel: numberInput("d-model"),
      layers: numberInput("layers"),
      learningRate: numberInput("learning-rate")
    };

    const tokenIds = state.tokenizer.encode(state.corpus);
    if (tokenIds.length < contextSize + 2) {
      throw new Error(`Need at least ${contextSize + 2} tokens after tokenization; add more training text or lower context.`);
    }

    state.model = new TinyGpt(config);
    const epochs = numberInput("epochs");
    const stepsPerEpoch = numberInput("steps");
    const startedAt = Date.now();
    const result = await state.model.train(tokenIds, {
      epochs,
      batchSize: numberInput("batch-size"),
      stepsPerEpoch,
      onProgress(update) {
        const done = ((update.epoch - 1) * stepsPerEpoch + update.step) / update.totalSteps;
        progressBar.style.width = `${Math.min(100, done * 100)}%`;
        trainingLog.textContent = `Epoch ${update.epoch}/${epochs}, step ${update.step}/${stepsPerEpoch}, loss ${update.loss.toFixed(4)} on ${update.backend}.`;
      }
    });

    const elapsedSeconds = ((Date.now() - startedAt) / 1000).toFixed(1);
    state.trained = true;
    state.lastTrainingSummary = {
      trained_at: new Date().toISOString(),
      backend: state.backend,
      final_loss: result.loss.toFixed(6),
      examples: String(result.examples),
      elapsed_seconds: elapsedSeconds
    };
    trainingLog.textContent = `Training complete in ${elapsedSeconds}s. Final loss ${result.loss.toFixed(4)}.`;
    chatState.textContent = "Ready";
    appendMessage("system", "Model is trained locally and ready to run.");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    trainingLog.textContent = message;
    chatState.textContent = "Training failed";
  } finally {
    setBusy(false);
  }
}

async function generate() {
  if (!state.model || !state.tokenizer || !state.trained) {
    return;
  }

  const prompt = promptInput.value.trim();
  if (!prompt) {
    return;
  }

  appendMessage("user", prompt);
  const assistantMessage = appendMessage("assistant", "");
  promptInput.value = "";
  setBusy(true);

  try {
    const promptIds = state.tokenizer.encode(prompt, true).slice(0, -1);
    const generated: number[] = [];
    await state.model.generate(
      promptIds,
      numberInput("max-new-tokens"),
      numberInput("temperature"),
      numberInput("top-k"),
      state.tokenizer.eosId,
      (tokenId) => {
        generated.push(tokenId);
        assistantMessage.textContent = state.tokenizer?.decode(generated) ?? "";
        chatLog.scrollTop = chatLog.scrollHeight;
      }
    );

    if (!assistantMessage.textContent) {
      assistantMessage.textContent = "(no continuation)";
    }
  } catch (error) {
    assistantMessage.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    setBusy(false);
  }
}

function downloadSafetensors() {
  if (!state.model || !state.tokenizer) {
    return;
  }

  const blob = createSafetensorsBlob(
    state.model.snapshotWeights(),
    state.model.config,
    state.tokenizer,
    state.lastTrainingSummary
  );
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `enkeltrain-${Date.now()}.safetensors`;
  anchor.click();
  URL.revokeObjectURL(url);
}

async function exportTrainingData(format: "zip" | "7z") {
  if (state.documents.length === 0) {
    return;
  }

  setBusy(true);
  const previousLog = trainingLog.textContent;
  trainingLog.textContent = `Creating ${format.toUpperCase()} training-data archive...`;
  try {
    const blob = format === "zip" ? await createTrainingDataZip(state.documents) : await createTrainingData7z(state.documents);
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `enkeltrain-training-data.${format}`;
    anchor.click();
    URL.revokeObjectURL(url);
    trainingLog.textContent = previousLog || "Training data exported.";
  } catch (error) {
    trainingLog.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    setBusy(false);
  }
}

fileInput.addEventListener("change", () => {
  if (fileInput.files) {
    void readFiles(fileInput.files);
  }
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  if (event.dataTransfer?.files) {
    void readFiles(event.dataTransfer.files);
  }
});

sampleButton.addEventListener("click", useSampleData);
exportZipButton.addEventListener("click", () => void exportTrainingData("zip"));
export7zButton.addEventListener("click", () => void exportTrainingData("7z"));
tokenizerButton.addEventListener("click", trainTokenizer);
trainButton.addEventListener("click", () => void trainModel());
generateButton.addEventListener("click", () => void generate());
downloadButton.addEventListener("click", downloadSafetensors);
promptInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    void generate();
  }
});

setBusy(false);
renderFiles();
initializeBestBackend().then((backend) => {
  state.backend = backend;
  renderBackend();
});
