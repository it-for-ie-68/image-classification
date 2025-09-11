import { Jimp } from "jimp";
import { Tensor } from "onnxruntime-web";
import { classList } from "./classes";
import { useEffect, useState } from "react";
import { load_model } from "./model";

interface Prediction {
  idx: number;
  probability: number;
  class: string;
}

function App() {
  const [session, setSession] = useState<any>(null);
  const [previewImage, setPreviewImage] = useState<any>(null);
  const [predictions, setPredictions] = useState<Prediction[] | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  useEffect(() => {
    setIsLoading(true);
    load_model().then((res) => {
      setSession(res);
      setIsLoading(false);
    });
  }, []);

  const handleSelectImage = (e: React.ChangeEvent<HTMLInputElement>) => {
    // User needs to select an image
    if (!e.target.files) return;
    // There has to be one file.
    if (e.target.files.length === 0) return;
    const file = e.target.files[0];

    // Read the file for preview
    const fileReaderPreview = new FileReader();
    fileReaderPreview.addEventListener("load", async (e) => {
      setPreviewImage(e.target?.result);
      setPredictions([]);
    });
    fileReaderPreview.readAsDataURL(file);

    // Read the file for ML
    const fileReaderML = new FileReader();
    fileReaderML.addEventListener("load", async (e) => {
      const image = await Jimp.fromBuffer(e.target?.result);
      const inputTensor = await processImageML(image);
      const feeds: any = {};
      feeds[session.inputNames[0]] = inputTensor;
      // Run inference
      const output = await session.run(feeds);
      // Format output
      const logits = output.class_logits.cpuData as number[];
      const _probs = softmax(logits);
      const probs = Array.prototype.slice.call(_probs); // Need to convert into regular array
      const tops = getTopClasses(probs, 5);
      setPredictions(tops);
    });
    fileReaderML.readAsArrayBuffer(file);
  };

  if (isLoading) return <div className="container">Loading...</div>;
  return (
    <div className="container">
      <h1>Image Classifier</h1>
      <div>
        <input type="file" onChange={handleSelectImage} />
      </div>
      <div>
        {previewImage && (
          <img
            id="my-img"
            src={previewImage}
            alt="preview-image"
            style={{ height: "50vh", borderRadius: "0.5rem" }}
          />
        )}
      </div>

      {predictions && (
        <article>
          <h2>Predictions</h2>
          {predictions?.map((p) => (
            <div key={p.class}>
              <b>{capital(p.class)}</b>
              &nbsp; &nbsp;
              <em>({(p.probability * 100).toFixed(0)}% likely)</em>
            </div>
          ))}
        </article>
      )}
    </div>
  );
}

export default App;

async function processImageML(
  image: any,
  dims: number[] = [1, 3, 224, 224],
  width: number = 224,
  height: number = 224
) {
  image.resize({
    w: width,
    h: height,
  });

  // 1. Get buffer data from image and create R, G, and B arrays.
  var imageBufferData = image.bitmap.data;
  const [redArray, greenArray, blueArray] = new Array(
    new Array<number>(),
    new Array<number>(),
    new Array<number>()
  );

  // 2. Loop through the image buffer and extract the R, G, and B channels
  for (let i = 0; i < imageBufferData.length; i += 4) {
    redArray.push(imageBufferData[i]);
    greenArray.push(imageBufferData[i + 1]);
    blueArray.push(imageBufferData[i + 2]);
    // skip data[i + 3] to filter out the alpha channel
  }

  // 3. Concatenate RGB to transpose [224, 224, 3] -> [3, 224, 224] to a number array
  const transposedData = redArray.concat(greenArray).concat(blueArray);

  // 4. convert to float32
  let i,
    l = transposedData.length; // length, we need this for the loop
  // create the Float32Array size 3 * 224 * 224 for these dimensions output
  const float32Data = new Float32Array(dims[1] * dims[2] * dims[3]);
  for (i = 0; i < l; i++) {
    float32Data[i] = transposedData[i] / 255.0; // convert to float
  }
  // 5. create the tensor object from onnxruntime-web.
  const inputTensor = new Tensor("float32", float32Data, dims);
  return inputTensor;
}

function softmax(logits: number[]): number[] {
  // For numerical stability, subtract the max logit before exponentiation
  const maxLogit = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sumExps);
}

function getTopClasses(probs: number[], n = 5) {
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
    class: classList[el.idx],
  }));

  return probIdxClass;
}

function capital(val: string) {
  return val.replace(
    /\w\S*/g,
    (text) => text.charAt(0).toUpperCase() + text.substring(1).toLowerCase()
  );
}
