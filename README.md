# EnkelTrain

EnkelTrain is a local-first TypeScript web app for training a tiny GPT-style causal language model in the browser. It uses TensorFlow.js and automatically prefers WebGPU when the browser supports it, with WebGL/CPU fallback.

## Run locally

```bash
npm install
npm run dev -- --port 5173
```

Open `http://127.0.0.1:5173/`.

## What it does

- Loads user-selected text-like files such as `.txt`, `.md`, `.csv`, `.json`, source code, and logs.
- Imports text files directly or from `.zip` / `.7z` archives.
- Exports the currently imported training data as `.zip` or `.7z`.
- Trains a byte-level GPT-style BPE tokenizer from scratch in the browser.
- Trains a small causal transformer language model locally in the browser.
- Runs local prompt continuation after training.
- Exports learned model weights as a `.safetensors` file with model config and tokenizer metadata embedded in the header.

## Practical limits

This is intentionally tiny compared with GPT-2 and trains from random initialization. Browser training is memory and compute constrained, so start with small corpora and defaults like a 32-token context, width 64, and one layer. Larger settings may be slow or fail depending on browser, GPU, and available memory.
