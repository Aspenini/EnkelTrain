import type { BpeTokenizer } from "./tokenizer";
import type { TinyGptConfig, WeightSnapshot } from "./tiny-gpt";

type SafetensorsTensor = {
  dtype: "F32";
  shape: number[];
  data_offsets: [number, number];
};

const textEncoder = new TextEncoder();

function writeUint64LE(target: Uint8Array, value: number, offset: number) {
  let remaining = BigInt(value);
  for (let i = 0; i < 8; i += 1) {
    target[offset + i] = Number(remaining & 0xffn);
    remaining >>= 8n;
  }
}

export function createSafetensorsBlob(
  weights: WeightSnapshot[],
  config: TinyGptConfig,
  tokenizer: BpeTokenizer,
  trainingSummary: Record<string, string>
): Blob {
  const header: Record<string, SafetensorsTensor | Record<string, string>> = {
    __metadata__: {
      format: "enkeltrain-tiny-gpt",
      model_config: JSON.stringify(config),
      tokenizer: JSON.stringify(tokenizer.toJson()),
      ...trainingSummary
    }
  };

  let cursor = 0;
  for (const weight of weights) {
    const byteLength = weight.values.byteLength;
    header[weight.name] = {
      dtype: "F32",
      shape: weight.shape,
      data_offsets: [cursor, cursor + byteLength]
    };
    cursor += byteLength;
  }

  let headerBytes = textEncoder.encode(JSON.stringify(header));
  const paddedHeaderLength = Math.ceil(headerBytes.length / 8) * 8;
  if (paddedHeaderLength !== headerBytes.length) {
    const padded = new Uint8Array(paddedHeaderLength);
    padded.set(headerBytes);
    padded.fill(0x20, headerBytes.length);
    headerBytes = padded;
  }

  const fileBytes = new Uint8Array(8 + headerBytes.length + cursor);
  writeUint64LE(fileBytes, headerBytes.length, 0);
  fileBytes.set(headerBytes, 8);

  let dataOffset = 8 + headerBytes.length;
  for (const weight of weights) {
    fileBytes.set(new Uint8Array(weight.values.buffer.slice(0)), dataOffset);
    dataOffset += weight.values.byteLength;
  }

  return new Blob([fileBytes], { type: "application/octet-stream" });
}
