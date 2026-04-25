import { unzipSync, zipSync } from "fflate";
import { Archive, ArchiveCompression, ArchiveFormat } from "libarchive.js";
import ArchiveWorker from "libarchive.js/dist/worker-bundle.js?worker";

Archive.init({
  getWorker: () => new ArchiveWorker()
});

export type TrainingDocument = {
  name: string;
  path: string;
  size: number;
  file: File;
};

const TEXT_EXTENSIONS =
  /\.(txt|md|markdown|csv|json|jsonl|html|xml|js|ts|tsx|jsx|py|rs|go|java|c|cpp|h|hpp|css|scss|sql|yaml|yml|toml|ini|log)$/i;

const ARCHIVE_EXTENSIONS = /\.(zip|7z)$/i;

export function looksTextBased(file: File | { name: string; type?: string }) {
  return file.type?.startsWith("text/") || file.type === "application/json" || TEXT_EXTENSIONS.test(file.name);
}

export function looksArchiveBased(file: File) {
  return ARCHIVE_EXTENSIONS.test(file.name);
}

export async function loadTrainingDocuments(files: FileList | File[]) {
  const documents: TrainingDocument[] = [];
  const rejected: string[] = [];

  for (const file of Array.from(files)) {
    if (looksArchiveBased(file)) {
      try {
        documents.push(...(await readArchive(file)));
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        rejected.push(`${file.name}: ${message}`);
      }
      continue;
    }

    if (looksTextBased(file)) {
      documents.push(toTrainingDocument(file, file.name));
    } else {
      rejected.push(file.name);
    }
  }

  return { documents, rejected };
}

export async function documentsToCorpus(documents: TrainingDocument[]) {
  const chunks: string[] = [];
  for (const document of documents) {
    chunks.push(await document.file.text());
  }
  return chunks.join("\n\n");
}

export async function createTrainingDataZip(documents: TrainingDocument[]) {
  const entries: Record<string, Uint8Array> = {};
  for (const document of documents) {
    entries[safeArchivePath(document.path)] = new Uint8Array(await document.file.arrayBuffer());
  }
  return new Blob([toArrayBuffer(zipSync(entries, { level: 6 }))], { type: "application/zip" });
}

export async function createTrainingData7z(documents: TrainingDocument[]) {
  const archiveFile = await Archive.write({
    files: documents.map((document) => ({
      file: document.file,
      pathname: safeArchivePath(document.path)
    })) as never,
    outputFileName: "enkeltrain-training-data.7z",
    compression: ArchiveCompression.LZMA,
    format: ArchiveFormat.SEVEN_ZIP,
    passphrase: null
  });
  return new Blob([archiveFile], { type: "application/x-7z-compressed" });
}

async function readArchive(file: File) {
  if (/\.zip$/i.test(file.name)) {
    return readZip(file);
  }
  return readLibarchive(file);
}

async function readZip(file: File) {
  const entries = unzipSync(new Uint8Array(await file.arrayBuffer()));
  const documents: TrainingDocument[] = [];
  for (const [path, bytes] of Object.entries(entries)) {
    if (path.endsWith("/") || !looksTextBased({ name: path })) {
      continue;
    }
    const safePath = safeArchivePath(path);
    documents.push(
      toTrainingDocument(
        new File([toArrayBuffer(bytes)], safePath, { type: guessMimeType(safePath), lastModified: file.lastModified }),
        safePath
      )
    );
  }
  return documents;
}

async function readLibarchive(file: File) {
  const archive = await Archive.open(file);
  const extracted = await archive.extractFiles();
  const documents: TrainingDocument[] = [];
  collectExtractedFiles(extracted, "", documents);
  return documents;
}

function collectExtractedFiles(value: unknown, path: string, documents: TrainingDocument[]) {
  if (value instanceof File) {
    const filePath = safeArchivePath(path || value.name);
    if (looksTextBased({ name: filePath, type: value.type })) {
      documents.push(toTrainingDocument(value, filePath));
    }
    return;
  }

  if (!value || typeof value !== "object") {
    return;
  }

  for (const [name, child] of Object.entries(value)) {
    collectExtractedFiles(child, path ? `${path}/${name}` : name, documents);
  }
}

function toTrainingDocument(file: File, path: string): TrainingDocument {
  const safePath = safeArchivePath(path);
  return {
    name: safePath.split("/").pop() || safePath,
    path: safePath,
    size: file.size,
    file
  };
}

function safeArchivePath(path: string) {
  const cleaned = path
    .replace(/\\/g, "/")
    .split("/")
    .filter((part) => part && part !== "." && part !== "..")
    .join("/");
  return cleaned || "training-data.txt";
}

function guessMimeType(path: string) {
  if (/\.(json|jsonl)$/i.test(path)) {
    return "application/json";
  }
  if (/\.(html|xml)$/i.test(path)) {
    return "text/html";
  }
  return "text/plain";
}

function toArrayBuffer(bytes: Uint8Array) {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}
