import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu";
import "@tensorflow/tfjs-backend-webgl";

export type TinyGptConfig = {
  vocabSize: number;
  contextSize: number;
  dModel: number;
  layers: number;
  learningRate: number;
};

export type TrainOptions = {
  epochs: number;
  batchSize: number;
  stepsPerEpoch: number;
  onProgress: (update: TrainingUpdate) => void;
};

export type TrainingUpdate = {
  epoch: number;
  step: number;
  totalSteps: number;
  loss: number;
  backend: string;
};

export type WeightSnapshot = {
  name: string;
  shape: number[];
  values: Float32Array;
};

type TransformerLayer = {
  ln1Gain: tf.Variable;
  ln1Bias: tf.Variable;
  ln2Gain: tf.Variable;
  ln2Bias: tf.Variable;
  wq: tf.Variable;
  wk: tf.Variable;
  wv: tf.Variable;
  wo: tf.Variable;
  ff1w: tf.Variable;
  ff1b: tf.Variable;
  ff2w: tf.Variable;
  ff2b: tf.Variable;
};

const variableRegistry: Array<{ name: string; variable: tf.Variable }> = [];

function register(name: string, variable: tf.Variable) {
  variableRegistry.push({ name, variable });
  return variable;
}

function randomVariable(name: string, shape: number[], scale = 0.02) {
  return register(name, tf.variable(tf.randomNormal(shape, 0, scale), true, name));
}

function zeroVariable(name: string, shape: number[]) {
  return register(name, tf.variable(tf.zeros(shape), true, name));
}

function oneVariable(name: string, shape: number[]) {
  return register(name, tf.variable(tf.ones(shape), true, name));
}

function layerNorm(x: tf.Tensor, gain: tf.Tensor, bias: tf.Tensor) {
  const mean = x.mean(-1, true);
  const variance = x.sub(mean).square().mean(-1, true);
  return x.sub(mean).mul(variance.add(1e-5).rsqrt()).mul(gain).add(bias);
}

function gelu(x: tf.Tensor) {
  return tf.tidy(() => {
    const coeff = tf.scalar(Math.sqrt(2 / Math.PI));
    return x.mul(0.5).mul(tf.scalar(1).add(tf.tanh(x.add(x.pow(tf.scalar(3)).mul(0.044715)).mul(coeff))));
  });
}

function linear3d(x: tf.Tensor, weights: tf.Tensor, bias?: tf.Tensor) {
  const [batchSize, sequenceLength] = x.shape;
  const [inputSize, outputSize] = weights.shape;
  if (!inputSize || !outputSize) {
    throw new Error("linear3d expects rank-2 weights.");
  }
  let y = x.reshape([-1, inputSize]).matMul(weights).reshape([batchSize, sequenceLength, outputSize]);
  if (bias) {
    y = y.add(bias);
  }
  return y;
}

function sampleFromLogits(logits: Float32Array, temperature: number, topK: number) {
  const adjusted = Array.from(logits).map((value, id) => ({
    id,
    value: value / Math.max(temperature, 0.05)
  }));
  adjusted.sort((a, b) => b.value - a.value);
  const candidates = adjusted.slice(0, Math.max(1, topK));
  const maxValue = candidates[0].value;
  const expValues = candidates.map((candidate) => Math.exp(candidate.value - maxValue));
  const total = expValues.reduce((sum, value) => sum + value, 0);
  let draw = Math.random() * total;
  for (let i = 0; i < candidates.length; i += 1) {
    draw -= expValues[i];
    if (draw <= 0) {
      return candidates[i].id;
    }
  }
  return candidates[candidates.length - 1].id;
}

export async function initializeBestBackend() {
  const hasWebGpu = "gpu" in navigator;
  const preferred = hasWebGpu ? ["webgpu", "webgl", "cpu"] : ["webgl", "cpu"];
  for (const backend of preferred) {
    try {
      await tf.setBackend(backend);
      await tf.ready();
      return tf.getBackend();
    } catch {
      continue;
    }
  }
  await tf.setBackend("cpu");
  await tf.ready();
  return tf.getBackend();
}

export class TinyGpt {
  readonly config: TinyGptConfig;
  private readonly tokenEmbedding: tf.Variable;
  private readonly positionEmbedding: tf.Variable;
  private readonly outputBias: tf.Variable;
  private readonly layers: TransformerLayer[] = [];
  private readonly optimizer: tf.Optimizer;

  constructor(config: TinyGptConfig) {
    this.config = config;
    variableRegistry.length = 0;
    this.tokenEmbedding = randomVariable("token_embedding", [config.vocabSize, config.dModel]);
    this.positionEmbedding = randomVariable("position_embedding", [config.contextSize, config.dModel]);
    this.outputBias = zeroVariable("output_bias", [config.vocabSize]);

    for (let index = 0; index < config.layers; index += 1) {
      const prefix = `layers.${index}`;
      this.layers.push({
        ln1Gain: oneVariable(`${prefix}.ln1_gain`, [config.dModel]),
        ln1Bias: zeroVariable(`${prefix}.ln1_bias`, [config.dModel]),
        ln2Gain: oneVariable(`${prefix}.ln2_gain`, [config.dModel]),
        ln2Bias: zeroVariable(`${prefix}.ln2_bias`, [config.dModel]),
        wq: randomVariable(`${prefix}.attention.wq`, [config.dModel, config.dModel]),
        wk: randomVariable(`${prefix}.attention.wk`, [config.dModel, config.dModel]),
        wv: randomVariable(`${prefix}.attention.wv`, [config.dModel, config.dModel]),
        wo: randomVariable(`${prefix}.attention.wo`, [config.dModel, config.dModel]),
        ff1w: randomVariable(`${prefix}.feed_forward.w1`, [config.dModel, config.dModel * 4]),
        ff1b: zeroVariable(`${prefix}.feed_forward.b1`, [config.dModel * 4]),
        ff2w: randomVariable(`${prefix}.feed_forward.w2`, [config.dModel * 4, config.dModel]),
        ff2b: zeroVariable(`${prefix}.feed_forward.b2`, [config.dModel])
      });
    }

    this.optimizer = tf.train.adam(config.learningRate);
  }

  forward(inputIds: tf.Tensor2D) {
    return tf.tidy(() => {
      const [, sequenceLength] = inputIds.shape;
      const tokenVectors = tf.gather(this.tokenEmbedding, inputIds);
      const positions = tf.range(0, sequenceLength, 1, "int32");
      const positionVectors = tf.gather(this.positionEmbedding, positions).expandDims(0);
      let x = tokenVectors.add(positionVectors);
      const causalMask = tf.linalg
        .bandPart(tf.ones([sequenceLength, sequenceLength]), -1, 0)
        .sub(1)
        .mul(1e9)
        .expandDims(0);

      for (const layer of this.layers) {
        const normed = layerNorm(x, layer.ln1Gain, layer.ln1Bias);
        const q = linear3d(normed, layer.wq);
        const k = linear3d(normed, layer.wk);
        const v = linear3d(normed, layer.wv);
        const scores = q.matMul(k, false, true).div(Math.sqrt(this.config.dModel)).add(causalMask);
        const attention = linear3d(tf.softmax(scores, -1).matMul(v), layer.wo);
        x = x.add(attention);

        const ffInput = layerNorm(x, layer.ln2Gain, layer.ln2Bias);
        const ff = linear3d(gelu(linear3d(ffInput, layer.ff1w, layer.ff1b)), layer.ff2w, layer.ff2b);
        x = x.add(ff);
      }

      const flat = x.reshape([-1, this.config.dModel]);
      return flat.matMul(this.tokenEmbedding, false, true).add(this.outputBias).reshape([
        inputIds.shape[0],
        sequenceLength,
        this.config.vocabSize
      ]);
    });
  }

  trainStep(inputBatch: number[][], targetBatch: number[][]) {
    const lossTensor = this.optimizer.minimize(() => {
      const input = tf.tensor2d(inputBatch, [inputBatch.length, this.config.contextSize], "int32");
      const target = tf.tensor2d(targetBatch, [targetBatch.length, this.config.contextSize], "int32");
      const logits = this.forward(input).reshape([-1, this.config.vocabSize]);
      const labels = tf.oneHot(target.flatten(), this.config.vocabSize);
      return tf.losses.softmaxCrossEntropy(labels, logits).mean();
    }, true) as tf.Scalar;

    const loss = lossTensor.dataSync()[0];
    lossTensor.dispose();
    return loss;
  }

  async train(tokenIds: number[], options: TrainOptions) {
    const examples = makeTrainingExamples(tokenIds, this.config.contextSize);
    const totalSteps = options.epochs * options.stepsPerEpoch;
    let lastLoss = Number.NaN;

    for (let epoch = 1; epoch <= options.epochs; epoch += 1) {
      for (let step = 1; step <= options.stepsPerEpoch; step += 1) {
        const batch = sampleBatch(examples, options.batchSize);
        lastLoss = this.trainStep(batch.inputs, batch.targets);
        options.onProgress({
          epoch,
          step,
          totalSteps,
          loss: lastLoss,
          backend: tf.getBackend()
        });

        if (step % 4 === 0) {
          await tf.nextFrame();
        }
      }
    }

    return { loss: lastLoss, examples: examples.length };
  }

  async generate(
    promptIds: number[],
    maxNewTokens: number,
    temperature: number,
    topK: number,
    eosId: number,
    onToken?: (tokenId: number) => void
  ) {
    const ids = [...promptIds];
    for (let i = 0; i < maxNewTokens; i += 1) {
      const context = ids.slice(-this.config.contextSize);
      while (context.length < this.config.contextSize) {
        context.unshift(0);
      }

      const nextId = tf.tidy(() => {
        const input = tf.tensor2d([context], [1, this.config.contextSize], "int32");
        const logits = this.forward(input);
        const lastLogits = logits.slice([0, this.config.contextSize - 1, 0], [1, 1, this.config.vocabSize]).reshape([
          this.config.vocabSize
        ]);
        const values = lastLogits.dataSync() as Float32Array;
        return sampleFromLogits(values, temperature, topK);
      });

      if (nextId === eosId) {
        break;
      }

      ids.push(nextId);
      onToken?.(nextId);
      await tf.nextFrame();
    }
    return ids;
  }

  snapshotWeights(): WeightSnapshot[] {
    return variableRegistry.map(({ name, variable }) => ({
      name,
      shape: variable.shape,
      values: new Float32Array(variable.dataSync())
    }));
  }
}

function makeTrainingExamples(tokenIds: number[], contextSize: number) {
  const examples: Array<{ input: number[]; target: number[] }> = [];
  for (let start = 0; start < tokenIds.length - contextSize - 1; start += 1) {
    examples.push({
      input: tokenIds.slice(start, start + contextSize),
      target: tokenIds.slice(start + 1, start + contextSize + 1)
    });
  }
  return examples;
}

function sampleBatch(examples: Array<{ input: number[]; target: number[] }>, batchSize: number) {
  const inputs: number[][] = [];
  const targets: number[][] = [];
  for (let i = 0; i < batchSize; i += 1) {
    const example = examples[Math.floor(Math.random() * examples.length)];
    inputs.push(example.input);
    targets.push(example.target);
  }
  return { inputs, targets };
}
