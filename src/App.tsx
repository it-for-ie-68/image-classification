import { Jimp, ResizeStrategy } from "jimp";
import { Tensor } from "onnxruntime-web";

import { useEffect, useState } from "react";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
import { load_model } from "./model";
function App() {
  const [session, setSession] = useState<any | null>(null);
  const [previewImage, setPreviewImage] = useState<any>(null);
  const [predictions, setPredictions] = useState<any[] | null>(null);
  const [ready, setReady] = useState<boolean>(false);
  useEffect(() => {
    load_model().then((res) => {
      console.log(res);
      setSession(res);
    });
  }, []);

  const handleSelectImage = (e: React.ChangeEvent<HTMLInputElement>) => {
    // User needs to select an image
    if (!e.target.files) return;
    // There has to be one file.
    if (e.target.files.length === 0) return;
    const file = e.target.files[0];

    console.log(e);
    loadImagefromPath(file.name);

    // Read the file
    const fileReader = new FileReader();
    fileReader.addEventListener("load", async () => {
      setPreviewImage(fileReader.result);
      setPredictions([]);
      console.log(fileReader);
    });
    fileReader.readAsDataURL(file);
  };

  function handleLoad(e: any) {
    if (!session) return;
  }

  return (
    <>
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
            style={{ height: "50vh" }}
            onLoad={handleLoad}
          />
        )}
      </div>

      <div>
        <h2>Prediction</h2>
        {predictions?.map((p) => (
          <div key={p.className}>
            {p.className} (มั่นใจ {(p.probability * 100).toFixed(0)}%)
          </div>
        ))}
      </div>
    </>
  );
}

export default App;

function imageDataToTensor(image: any, dims: number[]): Tensor {
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

async function loadImagefromPath(
  path: string,
  width: number = 224,
  height: number = 224
): Promise<any> {
  // Load image and resize
  const image = await Jimp.read(path);
  image.resize({
    w: width,
    h: height,
  });
  return image;
}
