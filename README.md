# Ausculta V143 Android / TensorFlow Lite handoff

This repository packages the selected **experimental V143** ensemble as one offline `.tflite` audio model, with Kotlin integration and Android Studio instructions. Allowed outputs are **Asthma, Pneumonia, Healthy, Bronchitis**, in exactly that order. No COPD output or training recordings are included.

**The model failed the original disease acceptance checks.** Conversion parity does not make it validated. Use an experimental label in the prototype UI. Model scores are uncalibrated and are not a recording's accuracy or a medically validated confidence estimate.

## Files your Android team needs

| File | Use |
|---|---|
| `ausculta/src/main/assets/ausculta/ausculta_v143_audio.tflite` | Complete audio preprocessing, normalization, three-parent ensemble and three-window pooling. |
| `ausculta/src/main/assets/ausculta/model_metadata.json` | SHA-256, labels, input/output contract and actual validation results. |
| `ausculta/src/main/java/com/ausculta/model/AuscultaModel.kt` | Kotlin loader, checksum/shape checks, prediction and audio quality checks. |
| `ausculta/src/main/java/com/ausculta/model/AudioWindows.kt` | Correct PCM normalization and first/middle/last crop selection. |
| `ausculta/src/main/java/com/ausculta/model/WavPcm16.kt` | Optional strict mono8kHz PCM16WAV decoder; handles metadata chunks. |
| `ausculta/src/main/assets/ausculta/golden_*` | Synthetic runtime test fixture; contains no patient recording. |
| `ausculta/src/androidTest/.../GoldenInferenceTest.kt` | Run on an emulator/device before accepting the integration. |
| `validation/` | Export parity and original held-out metrics. |

No `.npy` scaler, Python interpreter, sklearn model, external feature extractor, server or internet permission is needed by Android inference. The exported graph contains all scalers and model weights. `reference/` is export provenance for the original research checkout, not a standalone training kit; Android does not execute it.

## Android Studio integration

### Option A: copy into your existing app

1. Clone this repository, or download its ZIP.
2. Copy the whole `ausculta/src/main/assets/ausculta/` folder to your app's `app/src/main/assets/ausculta/`.
3. Copy the three Kotlin files into `app/src/main/java/com/ausculta/model/`. Keeping their declared package allows `import com.ausculta.model.AuscultaModel` from your UI.
4. Add `google()` and `mavenCentral()` to the repositories in your project's `settings.gradle.kts`.
5. Add this to **app-level** `build.gradle.kts`, then press **Sync Project with Gradle Files**:

```kotlin
dependencies {
    implementation("com.google.ai.edge.litert:litert:1.4.2")
}
android {
    // Keep your existing compileSdk. This package's standalone build uses 36.
    defaultConfig { minSdk = 24 }
    androidResources { noCompress += listOf("tflite", "f32") }
}
```

LiteRT is the current name for TensorFlow Lite and still loads `.tflite` files through `org.tensorflow.lite.Interpreter`. Use this single runtime dependency; remove conflicting old `org.tensorflow:tensorflow-lite` dependencies if they cause duplicate classes. This package uses the CPU Interpreter with XNNPACK disabled to match the verified portable runtime. Do not enable GPU/NNAPI or quantize without a new parity check.

### Option B: use the Android library module

The ready-built library is `dist/ausculta-v143-prototype.aar`; its SHA-256 and build/test status are in `validation/build_status.json`. You can use that AAR directly with the dependency instructions below. Rebuild from source if your team changes the integration code.

Open this repository in Android Studio (JDK17+, SDK36). The standalone build uses Gradle9.2.1 and Android Gradle Plugin9.0.1 with built-in Kotlin. Build:

```powershell
.\gradlew.bat :ausculta:assembleRelease :ausculta:testDebugUnitTest
```

The resulting AAR is `ausculta/build/outputs/aar/ausculta-release.aar`. Copy it into your app's `app/libs/`, then add:

```kotlin
dependencies {
    implementation(files("libs/ausculta-release.aar"))
    implementation("com.google.ai.edge.litert:litert:1.4.2")
}
```

A local AAR does not bring its Maven dependencies automatically; keep the LiteRT dependency. Alternatively import the `ausculta` module into your existing project and use `implementation(project(":ausculta"))`, adapting its Gradle plugin setup to your app. Option A avoids requiring your existing app to adopt this standalone build's AGP version.

## Call from Kotlin

Create one model instance on your background inference worker and reuse it:

```kotlin
import com.ausculta.model.AuscultaModel

// Run construction and analysis on your app's background worker, not the UI thread.
val model = AuscultaModel(applicationContext)
val result = model.analyzePcm16(recordedPcm16, sampleRate = 8000)

if (!result.quality.valid) {
    // Update UI on the main thread: show errors and let the user record again.
    val message = result.quality.errors.joinToString("\n")
} else {
    val analysis = result.analysisResult
    val modelScorePercent = result.modelScore!! * 100f
    val allClassScores = result.classScores
    val datasetAccuracyPercent = result.validation.canonicalArtifactAccuracy * 100.0
    // Show "Experimental result", "Model score (uncalibrated)" and
    // "Held-out dataset accuracy" as separate fields.
}
// Close after pending analysis finishes, when this worker/model is no longer used.
model.close()
```

`recordedPcm16` is a `ShortArray` of **decoded raw mono PCM samples**, not WAV file bytes, an AAC/M4A stream or MP3 data. The helper divides signed16-bit PCM by32768. If your input is already a normalized `FloatArray`, use `model.analyze(samples, 8000)`. Methods are synchronized because a single Interpreter must not run concurrently. Lifecycle, microphone permission, recording controls and rendering belong to the host app; the library does not open a microphone or create a UI.

For an existing mono8kHz PCM16WAV, use the optional decoder on a background worker:

```kotlin
import com.ausculta.model.WavPcm16
val decoded = WavPcm16.decode(wavBytes) // e.g. from your app's recording file
val result = model.analyzePcm16(decoded.samples, decoded.sampleRate)
```

The decoder rejects stereo, compressed audio, other sample rates and other bit depths with `IllegalArgumentException`; show that format error in the UI. It handles optional WAV metadata chunks instead of assuming every file has a44byte header. Keep audio files bounded to your intended recording duration when reading them into memory.

## Audio input: follow this exactly

- Actual sample rate: **8000Hz**, mono. Record at least5seconds;20seconds is a useful prototype capture duration, not an accuracy guarantee.
- Amplitude: finite float32 PCM in[-1,1]. No peak normalization, new denoising, bandpass, gain augmentation or handmade spectrogram.
- The wrapper selects **first, middle, last**5second crops. The model receives `[3,40000]` float32 values and produces `[1,4]` float32 scores.
- The graph implements the original512FFT/128hop/64mel frontend, full-time64bin compression,572descriptors, existing per-parent normalization and fixed ensemble blend. Do not apply those steps again outside the model.
- `AudioRecord.getMinBufferSize(8000, CHANNEL_IN_MONO, ENCODING_PCM_16BIT)` and recorder state must be checked. If8kHz capture is unsupported, correctly resample with an anti-aliasing implementation and validate it; changing the sample-rate label or keeping every nth sample is incorrect. The provided helper intentionally rejects other rates instead of silently using a different resampler.
- For vest/Bluetooth/USB audio, decode the real device format first. Passing phone microphone samples does not establish performance on the vest, and vice versa.
- Silence and malformed/short inputs produce no diagnosis or score. Weak/clipped signals produce engineering warnings, not clinical quality guarantees.

For phone-microphone capture, the **host app** needs:

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

Request that permission at runtime before using `AudioRecord`, handle partial reads and negative read errors, and stop/release the recorder on cancellation. Configure `MediaRecorder.AudioSource.UNPROCESSED` only when the device reports support; otherwise document the capture path. The wrapper takes the final decoded sample buffer; it does not manage recording permissions.

## What the scores mean

| UI item | Kotlin value | Meaning |
|---|---|---|
| Analysis result | `analysisResult` | Highest-scoring allowed diagnosis, experimental. |
| Model score | `modelScore`, `classScores` | Uncalibrated outputs; not an accuracy percentage for this recording. |
| Accuracy information | `validation.canonicalArtifactAccuracy` | This fixed artifact's held-out recording accuracy:59.78%. |
| Configuration comparison | `validation.configurationMeanAccuracy` | Mean across9research folds:62.20%; not this artifact's score. |
| Prototype status | `experimental`, `acceptancePassed`, `confidenceCalibrated` | true,false,false. |

The configuration balanced score is0.562053 (mean recording/patient macroF1), with5/9strict passes and3/9collapse folds. The canonical artifact has balanced0.512087 and only15.38%Asthma patient recall, so original acceptance remains failed. There is no calibrated confidence score. Do not replace the validation accuracy with the current maximum model score.

## Verify the integration

1. Build the library and run its JVM crop/PCM tests.
2. Run the synthetic golden test on an emulator or physical device:

```powershell
.\gradlew.bat :ausculta:connectedDebugAndroidTest
```

If copying files into another app, also copy the instrumentation test to that app's `src/androidTest` and add AndroidX test runner/JUnit dependencies. The golden test compares each output against `golden_output.json` within1e-4 and checks that silence produces no result. It is a runtime contract test, not evidence of disease accuracy.

The Python export comparison covers all450original canonical held-out recordings and their patients, with zero recording/patient argmax changes. See `validation/conversion_report.json` for numerical differences and unchanged disease metrics. Android execution has a separate status in `validation/build_status.json`; do not infer device validation from desktop conversion success. Repeat the golden test on your team's actual runtime/device after dependency, delegate or preprocessing changes.

Optional desktop verification, with Python3.11:

```powershell
python -m pip install -r verification/requirements.txt
python verification/verify_model.py
```

The model was exported with built-in TFLite operators only; no Select TF Ops/Flex runtime is required. `SHA256SUMS.json` lists the packaged files' checksums.

## Publishing to the specified repository

Target: <https://github.com/shinee-devs/Ausculta-AI-Model-10-17-26->.

If upload is pending because the connected account lacks write access, the repository owner can upload this package through **Add file > Upload files**, or run these commands from this package folder after reviewing the files:

```powershell
git init -b main
git add .
git commit -m "Add experimental V143 TFLite model and Android integration"
git remote add origin https://github.com/shinee-devs/Ausculta-AI-Model-10-17-26-.git
git push -u origin main
```

Use the owner's normal GitHub login/credential manager. Do not place tokens or passwords into files or commit messages. If this folder already contains a Git repository or an existing remote, inspect it and omit duplicate initialization/remote commands. No force push is needed for the currently empty target.

## Official references

- [LiteRT migration and Interpreter compatibility](https://ai.google.dev/edge/litert/migration)
- [LiteRT Android runtime versions](https://ai.google.dev/edge/litert/android)
- [Android AudioRecord API](https://developer.android.com/reference/android/media/AudioRecord)
- [Android Gradle Plugin9.0.1 and built-in Kotlin](https://developer.android.com/build/releases/agp-9-0-0-release-notes)

This package adds no new training or disease approval. No original datasets, patient manifests, user/manikin recordings, historical models or credentials belong in this repository.
