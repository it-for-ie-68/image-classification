import { classList } from "./classes";
import { Tensor } from "onnxruntime-web";

export async function convertImageToTensor(
  image: any,
  dims: number[] = [1, 3, 224, 224],
  width: number = 224,
  height: number = 224
) {
  // Resize image in-place as needed
  image.resize({ w: width, h: height });

  const { data } = image.bitmap; // Uint8ClampedArray, [R,G,B,A,...]
  const [_, channels, imgH, imgW] = dims;
  const float32Data = new Float32Array(channels * imgH * imgW);

  // Pack RGB into channels-first (R:0, G:1, B:2), normalized to [0,1]
  let pixelIndex = 0;
  for (let y = 0; y < imgH; y++) {
    for (let x = 0; x < imgW; x++) {
      const idx = (y * imgW + x) * 4;
      float32Data[0 * imgH * imgW + y * imgW + x] = data[idx] / 255.0; // R
      float32Data[1 * imgH * imgW + y * imgW + x] = data[idx + 1] / 255.0; // G
      float32Data[2 * imgH * imgW + y * imgW + x] = data[idx + 2] / 255.0; // B
      pixelIndex++;
    }
  }

  // Shape: [1, 3, height, width] for ONNX Runtime Web
  return new Tensor("float32", float32Data, dims);
}

export interface Output {
  class_logits: ClassLogits;
}

export interface ClassLogits {
  cpuData: { [key: string]: number };
  dataLocation: string;
  type: string;
  dims: number[];
  size: number;
}

export function formatOutput(output: Output) {
  const _logits = output.class_logits.cpuData; // FloatArray
  const logits = Array.prototype.slice.call(_logits) as number[]; // Convert to Array
  const probs = softmax(logits);
  const tops = getTopClasses(probs, 5);
  return tops;
}

export function softmax(logits: number[]): number[] {
  // For numerical stability, subtract the max logit before exponentiation
  const maxLogit = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sumExps);
}

export interface Prediction {
  idx: number;
  probability: number;
  class: string;
}

export function getTopClasses(probs: number[], n = 5): Prediction[] {
  // Pair each probability with its index
  const probIdx = probs.map((prob, idx) => ({
    idx,
    probability: prob,
  }));

  // Sort by probability descending
  probIdx.sort((a, b) => b.probability - a.probability);
  // Get top n
  const probIdxSlice = probIdx.slice(0, n);
  // Add class
  const probIdxClass = probIdxSlice.map((el) => ({
    ...el,
    class: capital(classList[el.idx]),
  }));

  return probIdxClass;
}

export function capital(val: string) {
  return val.replace(
    /\w\S*/g,
    (text) => text.charAt(0).toUpperCase() + text.substring(1).toLowerCase()
  );
}
