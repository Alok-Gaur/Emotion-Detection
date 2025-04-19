<h1 align="center">🧠 Emotion Detection using Deep Learning</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" />
  <img src="https://img.shields.io/badge/Framework-TensorFlow-green.svg" />
  <img src="https://img.shields.io/badge/GUI-Tkinter-blueviolet" />
</p>

<p align="center">
  A deep learning project to detect human emotions (Happy, Sad, Angry) from images, built using TensorFlow, Custom CNNs, and a Tkinter GUI with emoji visualization.
</p>

<hr/>

<h2>📁 Project Structure</h2>

<pre>
Emotion Detection/
├── ai_model/
│   ├── data/
│   │   ├── Dataset URL.txt
│   │   ├── README.md
│   │   ├── test/
│   │   │   ├── angry/
│   │   │   ├── happy/
│   │   │   └── sad/
│   │   └── train/
│   │       ├── angry/
│   │       ├── happy/
│   │       └── sad/
│   ├── model/
│   │   └── emotion_detection_model4.keras
│   ├── prediction/
│   │   ├── prediction.py
│   └── training/
│       ├── dataset_visualization.py
│       ├── emotion_detection_model.py
│       ├── test.py
│       ├── train.py
│       └── util.py
├── configurations/
│   ├── .env
│   ├── config.py
│   └── git_token.txt
├── Emotion_Detection4.weights.h5
├── gui_main2.py
├── main.py
├── .gitignore
└── README.md
</pre>

<hr/>

<h2>🚀 Features</h2>

<ul>
  <li><strong>Train:</strong> Train a custom CNN model on emotion-labeled images.</li>
  <li><strong>Test:</strong> Evaluate the model on unseen images or folders.</li>
  <li><strong>Predict:</strong> Predict emotion on a single image or batch of images.</li>
  <li><strong>Batch Prediction:</strong> Save prediction results in a CSV file.</li>
  <li><strong>GUI:</strong> Step-by-step graphical interface using Tkinter with emoji results for emotions.</li>
</ul>

<hr/>

<h2>🛠️ Setup</h2>

<ol>
  <li>Clone the repository:
    <pre><code>git clone https://github.com/your-username/emotion-detection.git</code></pre>
  </li>
  <li>Install dependencies:
    <pre><code>pip install -r requirements.txt</code></pre>
  </li>
  <li>Add dataset:
    <p>Download and extract the dataset as instructed in <code>ai_model/data/Dataset URL.txt</code> into <code>ai_model/data/train</code> and <code>ai_model/data/test</code>.</p>
  </li>
</ol>

<hr/>

<h2>🧪 Run from Terminal</h2>

<pre>
python main.py --mode train --train_directory_path ai_model/data/train --val_directory_path ai_model/data/test

python main.py --mode test --directory_path ai_model/data/test --model_path ai_model/model/emotion_detection_model4.keras

python main.py --mode predict --image_path path/to/image.jpg --model_path ai_model/model/emotion_detection_model4.keras
</pre>

<hr/>

<h2>🖼️ Run GUI</h2>

<p>Launch the graphical interface:</p>

<pre><code>python gui_main2.py</code></pre>

<ul>
  <li>Step-by-step interface for Train, Test, and Predict modes.</li>
  <li>Live emoji output: 😊 😢 😠</li>
  <li>Batch prediction saves a <code>batch_predictions.csv</code> file with results.</li>
</ul>

<hr/>

<h2>📦 Model Output</h2>

<p>Models are saved as:</p>
<ul>
  <li><code>.keras</code> for TensorFlow SavedModels</li>
  <li><code>.h5</code> optionally for legacy support</li>
</ul>

<hr/>

<h2>📄 Output Example (Batch Prediction)</h2>

<pre>
image_path,prediction
ai_model/data/test/angry/img1.jpg,angry
ai_model/data/test/happy/img2.jpg,happy
</pre>

<hr/>

<h2>👤 Author</h2>

<p>Created by <strong>Your Name</strong></p>
<p>Feel free to connect on GitHub or LinkedIn!</p>

<hr/>

<h2>📜 License</h2>

<p>This project is licensed under the MIT License.</p>
