# Cacao pod classifier prototype

Streamlit prototype for the paired **three-class, image-level** models in our ACM paper: the Soh Table 4 VGG-style baseline and the same architecture with CBAM after each of its five pooling layers. A single uploaded or camera photo is resized to 224 × 224 RGB, scaled by 255, and passed to both trained models. The site compares their three class scores and displays their saved held-out test metrics.

The previous version of this repository was a four-class YOLO detector. Its `.tflite` files are incompatible with this app. Do not rename them to `baseline.tflite` or `cbam.tflite`.

## Current status

The new Kaggle export has been installed locally. Both model files passed SHA-256 checks against `model_manifest.json`, and their saved split signatures match. The two `.tflite` files are in `models/` and ignored by Git; the manifest is committed.

**Model quality warning:** Both saved runs predicted `healthy` for all 441 held-out test images. Each scored 75.96% accuracy because 335 of those images were healthy, but **Black Pod Rot recall and Pod Borer recall were both 0%**. CBAM did not improve these held-out predictions. The app displays this warning and must not be presented as reliable disease identification. A successful inference on the deployed site still needs to be checked with a real image before the showcase.

| Saved model | Test accuracy | Macro F1 | Healthy recall | Black Pod Rot recall | Pod Borer recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 75.96% | 0.288 | 100% | 0% | 0% |
| VGG + CBAM | 75.96% | 0.288 | 100% | 0% | 0% |

These are the metrics in the exported manifest, based on the held-out test set, not predictions made by this web app.

## Showcase input

1. Open the deployed app on the laptop or phone you will use. Have one clear cacao pod photo saved on that device as a backup.
2. Under **01 / Input photograph**, choose **Upload** and select that photo, or choose **Camera**, allow browser camera access, and take a photo. The camera belongs to the device opening the app, not the server.
3. Check the 224 × 224 preview, then click **Compare models**. The first run may take longer while the app downloads and loads both model files. Verify this entire flow before the presentation.
4. Explain the two predictions as research outputs. Both saved models predicted Healthy for every held-out test image, so the site must not be presented as a reliable disease diagnosis.

## Repeat the export after future retraining

1. In Kaggle, open the finished **ACM baseline** and **ACM CBAM** notebook outputs. Check that each has `final.weights.h5`, `run_config.json`, and `test_metrics.json` inside its `acm_defense_baseline/` or `acm_defense_cbam/` output folder. Check `completed_epochs` is `100` in both metrics files. The preparation notebook's manifest is already represented by the split signature in both runs; it is not another input to the exporter.
2. Import [export_for_streamlit.ipynb](export_for_streamlit.ipynb) as a **new Kaggle notebook**. Under **Add Input → Notebook**, attach the *saved outputs* of both completed training notebooks. The raw cacao image dataset is not needed for this export. Select a Kaggle GPU and **Run All**, then **Save Version → Save & Run All (Commit)** if you want a durable exported output.
3. From that export notebook's `/kaggle/working/streamlit_export/` output, download all three files: `baseline.tflite`, `cbam.tflite`, and `model_manifest.json`. The notebook reconstructs the exact layer names, loads the final weights, checks the two split signatures, checks each conversion against Keras predictions, and refuses to export a partial run. It does **not** retrain either model.
4. Copy `model_manifest.json` into this repository's root. For a **local** test, place the two `.tflite` files in `models/`. They are excluded from Git by `.gitignore`. If Kaggle gives you a ZIP, run `python tools/install_export_zip.py "C:\Users\rjmia\Downloads\results.zip"` from this repository after installing the requirements; it verifies all three files and installs them.
5. For Streamlit Community Cloud, upload both `.tflite` files as assets of a **public GitHub Release** for this repository, then configure the app's Streamlit secrets with their exact URLs:

   ```toml
   BASELINE_MODEL_URL = "https://github.com/rhyohie/Thesis_Prototype/releases/download/acm-models-v1/baseline.tflite"
   CBAM_MODEL_URL = "https://github.com/rhyohie/Thesis_Prototype/releases/download/acm-models-v1/cbam.tflite"
   ```

   Use the actual release tag in the URLs if it differs from `acm-models-v1`. The app checks each downloaded file's SHA-256 against `model_manifest.json` before loading it. This also allows local files to be tested without hosting them. The model files are hundreds of MB, so initial loading may take time and memory. Verify the deployed app with one image before using it for a presentation.

## Run locally on this laptop

Open **Command Prompt** and run:

```bat
cd /d "C:\Users\rjmia\Documents\Codex\2026-09-25\rea\Thesis_Prototype"
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

`py -3.12` requires Python 3.12 installed on the laptop. If `py` is unavailable, install Python 3.12 or invoke its full executable path. Open the local URL shown by Streamlit. The models are already installed under `models/`, so the page can display the actual held-out metrics.

On this laptop, a local smoke test showed that Windows Application Control blocks the native DLL in the LiteRT Python wheel. If the same policy applies when you add the models, the app will show a runtime error instead of a prediction. The Linux Streamlit deployment remains the intended inference target; it still needs to be tested with the real exported files.

## Review and push with Command Prompt

The repository is already cloned locally at the path above. After reviewing the model quality warning and setting up the GitHub Release assets and Streamlit secrets, use:

```bat
cd /d "C:\Users\rjmia\Documents\Codex\2026-09-25\rea\Thesis_Prototype"
git status
git add app.py model_runtime.py requirements.txt README.md .gitignore .streamlit models tools export_for_streamlit.ipynb model_manifest.json
git add -u
git commit -m "Rebuild cacao prototype for paired ACM classifiers"
git -c http.sslBackend=openssl push origin main
```

Check `git status` before committing. Neither `models/*.tflite`, `.venv`, nor `.streamlit/secrets.toml` should be committed. GitHub may ask you to sign in for the push. If you are working in a branch rather than `main`, push that branch and open a pull request instead of pushing `main` directly.

The old `.tflite` files should be removed from the new Git commit. If `git status` does not show them as deleted, use `git rm -- baseline_seed42_float32.tflite cbam_seed101_float32.tflite` before the commit. This does not erase Git history.

## Method contract

- Classes in order: `healthy`, `black_pod_rot`, `pod_borer`.
- Input: EXIF orientation correction, RGB conversion, 224 × 224 bilinear resize, float32 values divided by 255.
- Both models: 2–2–2–3–2 convolutions, five max-pooling layers, Flatten, Dense 4096 / Dense 4096 / Dense 3, random seeded initialization.
- CBAM model: channel then spatial attention after each pool; reduction ratio 16 and spatial kernel 7 in the trained notebook. Singh et al.'s cited CBAM-VGG16 study specifies a reduction ratio of 8, so this trained model is not an exact copy of that CBAM setting. Changing the ratio requires retraining and a new export.
- Training runs: the same prepared 3072 / 877 / 441 train / validation / test split, seed 42, Adamax at learning rate 0.1, batch size 64, 100 epochs, final-epoch weights. No augmentation, ImageNet weights, class weights, dropout, L2, or early stopping in the paper-parameter profile.
- The app is image classification only. It does not draw detection boxes or give a field diagnosis.

The paper names this reconstruction “VGG-16,” although its Table 4 explicitly lists 11 convolution layers. The prototype uses the table architecture implemented by the training notebooks, not the canonical 13-convolution ImageNet VGG-16.
