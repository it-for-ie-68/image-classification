import { Jimp } from "jimp";
import { useEffect, useState } from "react";
import { load_model } from "./model";
import { convertImageToTensor, formatOutput, type Prediction } from "./ml";

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

    // ---------- Generate Image Preview ----------
    const fileReaderPreview = new FileReader();
    fileReaderPreview.addEventListener("load", async (e) => {
      setPreviewImage(e.target?.result);
      // setPredictions([]);
    });
    fileReaderPreview.readAsDataURL(file); // Start reading the file

    // ---------- ML Inference ----------
    const fileReaderML = new FileReader();
    fileReaderML.addEventListener("load", async (e) => {
      const buffer = e.target?.result;
      if (!buffer || !(buffer instanceof ArrayBuffer)) return;
      const image = await Jimp.fromBuffer(buffer);
      const inputTensor = await convertImageToTensor(image);
      const feeds: any = {};
      feeds[session.inputNames[0]] = inputTensor;
      const output = await session.run(feeds); // Run inference
      const topClasses = formatOutput(output);
      console.log(topClasses);
      setPredictions(topClasses);
    });
    fileReaderML.readAsArrayBuffer(file); // Start reading the file
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
              <b>{p.class}</b>
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
