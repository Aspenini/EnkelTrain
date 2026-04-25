export type TokenizerJson = {
  kind: "byte-level-bpe";
  specialTokens: string[];
  merges: Array<[string, string]>;
  vocab: string[];
};

const SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"];
const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

const { byteToUnicode, unicodeToByte } = createByteUnicodeMaps();
const PRETOKEN_PATTERN =
  /'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export class BpeTokenizer {
  readonly kind = "byte-level-bpe";
  readonly specialTokens = SPECIAL_TOKENS;
  readonly merges: Array<[string, string]>;
  readonly vocab: string[];
  private readonly tokenToId: Map<string, number>;

  constructor(merges: Array<[string, string]>, vocab: string[]) {
    this.merges = merges;
    this.vocab = vocab;
    this.tokenToId = new Map(vocab.map((token, index) => [token, index]));
  }

  get padId() {
    return this.tokenToId.get("<pad>") ?? 0;
  }

  get bosId() {
    return this.tokenToId.get("<bos>") ?? 1;
  }

  get eosId() {
    return this.tokenToId.get("<eos>") ?? 2;
  }

  get unkId() {
    return this.tokenToId.get("<unk>") ?? 3;
  }

  encode(text: string, includeBounds = true): number[] {
    const ids: number[] = [];
    for (const pretoken of pretokenize(text)) {
      let pieces = Array.from(bytesToUnicodeString(pretoken));
      pieces = applyMerges(pieces, this.merges);
      for (const piece of pieces) {
        ids.push(this.tokenToId.get(piece) ?? this.unkId);
      }
    }

    return includeBounds ? [this.bosId, ...ids, this.eosId] : ids;
  }

  decode(ids: number[]): string {
    const bytes: number[] = [];
    for (const id of ids) {
      const token = this.vocab[id] ?? "";
      if (this.specialTokens.includes(token)) {
        continue;
      }
      for (const char of Array.from(token)) {
        const byte = unicodeToByte.get(char);
        if (byte !== undefined) {
          bytes.push(byte);
        }
      }
    }
    return textDecoder.decode(new Uint8Array(bytes));
  }

  toJson(): TokenizerJson {
    return {
      kind: this.kind,
      specialTokens: this.specialTokens,
      merges: this.merges,
      vocab: this.vocab
    };
  }

  static fromJson(json: TokenizerJson): BpeTokenizer {
    return new BpeTokenizer(json.merges, json.vocab);
  }
}

export type TokenizerTrainingStats = {
  vocabSize: number;
  mergeCount: number;
  characterCount: number;
  pretokenCount: number;
};

export function trainBpeTokenizer(text: string, targetVocabSize: number): {
  tokenizer: BpeTokenizer;
  stats: TokenizerTrainingStats;
} {
  const byteAlphabet = Array.from(byteToUnicode.values());
  const normalizedTarget = Math.max(targetVocabSize, SPECIAL_TOKENS.length + byteAlphabet.length);
  const vocab = [...SPECIAL_TOKENS, ...byteAlphabet];
  const vocabSet = new Set(vocab);
  const merges: Array<[string, string]> = [];
  let words = buildWordFrequencies(text);

  while (vocab.length < normalizedTarget) {
    const pairCounts = new Map<string, number>();
    for (const word of words) {
      for (let i = 0; i < word.symbols.length - 1; i += 1) {
        const key = pairKey(word.symbols[i], word.symbols[i + 1]);
        pairCounts.set(key, (pairCounts.get(key) ?? 0) + word.count);
      }
    }

    let bestPair: [string, string] | null = null;
    let bestCount = 1;
    for (const [key, count] of pairCounts) {
      if (count > bestCount) {
        const [left, right] = splitPairKey(key);
        const merged = left + right;
        if (!vocabSet.has(merged)) {
          bestPair = [left, right];
          bestCount = count;
        }
      }
    }

    if (!bestPair) {
      break;
    }

    const merged = bestPair[0] + bestPair[1];
    words = words.map((word) => ({
      count: word.count,
      symbols: mergePair(word.symbols, bestPair)
    }));
    merges.push(bestPair);
    vocab.push(merged);
    vocabSet.add(merged);
  }

  return {
    tokenizer: new BpeTokenizer(merges, vocab),
    stats: {
      vocabSize: vocab.length,
      mergeCount: merges.length,
      characterCount: text.length,
      pretokenCount: words.reduce((sum, word) => sum + word.count, 0)
    }
  };
}

function pretokenize(text: string) {
  return text.match(PRETOKEN_PATTERN) ?? [];
}

function buildWordFrequencies(text: string) {
  const counts = new Map<string, number>();
  for (const pretoken of pretokenize(text)) {
    const encoded = bytesToUnicodeString(pretoken);
    counts.set(encoded, (counts.get(encoded) ?? 0) + 1);
  }

  return Array.from(counts.entries()).map(([encoded, count]) => ({
    symbols: Array.from(encoded),
    count
  }));
}

function bytesToUnicodeString(text: string) {
  let output = "";
  for (const byte of textEncoder.encode(text)) {
    output += byteToUnicode.get(byte) ?? "";
  }
  return output;
}

function applyMerges(symbols: string[], merges: Array<[string, string]>) {
  let pieces = symbols;
  for (const merge of merges) {
    pieces = mergePair(pieces, merge);
  }
  return pieces;
}

function mergePair(symbols: string[], pair: [string, string]) {
  const [left, right] = pair;
  const merged = left + right;
  const next: string[] = [];
  for (let i = 0; i < symbols.length; i += 1) {
    if (symbols[i] === left && symbols[i + 1] === right) {
      next.push(merged);
      i += 1;
    } else {
      next.push(symbols[i]);
    }
  }
  return next;
}

function pairKey(left: string, right: string) {
  return `${left}\u0001${right}`;
}

function splitPairKey(key: string): [string, string] {
  const index = key.indexOf("\u0001");
  return [key.slice(0, index), key.slice(index + 1)];
}

function createByteUnicodeMaps() {
  const bytes = [
    ...range(33, 126),
    ...range(161, 172),
    ...range(174, 255)
  ];
  const chars = [...bytes];
  let next = 0;

  for (let byte = 0; byte < 256; byte += 1) {
    if (!bytes.includes(byte)) {
      bytes.push(byte);
      chars.push(256 + next);
      next += 1;
    }
  }

  const byteToUnicodeMap = new Map<number, string>();
  const unicodeToByteMap = new Map<string, number>();
  for (let index = 0; index < bytes.length; index += 1) {
    const char = String.fromCodePoint(chars[index]);
    byteToUnicodeMap.set(bytes[index], char);
    unicodeToByteMap.set(char, bytes[index]);
  }

  return {
    byteToUnicode: byteToUnicodeMap,
    unicodeToByte: unicodeToByteMap
  };
}

function range(start: number, endInclusive: number) {
  const values: number[] = [];
  for (let value = start; value <= endInclusive; value += 1) {
    values.push(value);
  }
  return values;
}
