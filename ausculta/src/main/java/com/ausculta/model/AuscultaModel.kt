package com.ausculta.model

import android.content.Context
import org.json.JSONObject
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.sqrt

data class AudioQuality(
    val valid: Boolean, val errors: List<String>, val warnings: List<String>,
    val durationSeconds: Double, val rms: Double, val clippedFraction: Double
)
data class ValidationPerformance(
    val configurationMeanAccuracy: Double,
    val canonicalArtifactAccuracy: Double,
    val strictPasses: Int, val collapseFolds: Int, val folds: Int
)
data class AuscultaResult(
    val analysisResult: String?, val modelScore: Float?, val classScores: Map<String, Float>,
    val quality: AudioQuality, val validation: ValidationPerformance,
    val experimental: Boolean = true, val confidenceCalibrated: Boolean = false,
    val acceptancePassed: Boolean = false
)

/** CPU-only prototype. Construct once; analyze on a worker thread; close on teardown. */
class AuscultaModel(context: Context) : Closeable {
    companion object {
        val CLASS_ORDER = listOf("Asthma", "Pneumonia", "Healthy", "Bronchitis")
        private const val ASSET_ROOT = "ausculta/"
    }
    private val interpreter: Interpreter
    val validation: ValidationPerformance
    init {
        val metadata = JSONObject(context.assets.open(ASSET_ROOT + "model_metadata.json").bufferedReader().use { it.readText() })
        require(metadata.getBoolean("experimental") && !metadata.getBoolean("acceptance_passed"))
        val labels = metadata.getJSONArray("class_order")
        require((0 until labels.length()).map { labels.getString(it) } == CLASS_ORDER)
        val bytes = context.assets.open(ASSET_ROOT + "ausculta_v143_audio.tflite").use { it.readBytes() }
        val digest = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it.toInt() and 255) }
        require(digest == metadata.getString("model_sha256")) { "Model checksum mismatch" }
        val buffer = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).put(bytes)
        buffer.rewind()
        interpreter = Interpreter(buffer, Interpreter.Options().setNumThreads(2).setUseXNNPACK(false))
        require(interpreter.getInputTensor(0).shape().contentEquals(intArrayOf(3, 40000)))
        require(interpreter.getOutputTensor(0).shape().contentEquals(intArrayOf(1, 4)))
        require(interpreter.getInputTensor(0).dataType() == DataType.FLOAT32)
        require(interpreter.getOutputTensor(0).dataType() == DataType.FLOAT32)
        val cv = metadata.getJSONObject("configuration_cross_validation")
        validation = ValidationPerformance(cv.getDouble("recording_accuracy"),
            metadata.getJSONObject("artifact_fold_performance").getDouble("recording_accuracy"),
            cv.getInt("strict_passes"), cv.getInt("collapse_folds"), cv.getInt("folds"))
    }
    @Synchronized
    fun analyzePcm16(samples: ShortArray, sampleRate: Int): AuscultaResult =
        analyze(AudioWindows.normalizePcm16(samples), sampleRate)

    @Synchronized
    fun analyze(samples: FloatArray, sampleRate: Int): AuscultaResult {
        val quality = inspectQuality(samples, sampleRate)
        if (!quality.valid) return AuscultaResult(null, null, emptyMap(), quality, validation)
        val scores = runWindows(AudioWindows.fromRecording(samples, sampleRate))
        val best = scores.indices.maxByOrNull { scores[it] }!!
        return AuscultaResult(CLASS_ORDER[best], scores[best], CLASS_ORDER.zip(scores.toList()).toMap(), quality, validation)
    }
    /** Test hook for the synthetic runtime fixture, never bypass quality in the app UI. */
    @Synchronized
    internal fun runWindows(windows: Array<FloatArray>): FloatArray {
        require(windows.size == 3 && windows.all { it.size == 40000 && it.all { v -> v.isFinite() && v >= -1f && v <= 1f } })
        val input = ByteBuffer.allocateDirect(3 * 40000 * 4).order(ByteOrder.nativeOrder())
        windows.forEach { view -> view.forEach { input.putFloat(it) } }; input.rewind()
        val output = ByteBuffer.allocateDirect(4 * 4).order(ByteOrder.nativeOrder())
        interpreter.run(input, output); output.rewind()
        val scores = FloatArray(4) { output.float }
        check(scores.all { it.isFinite() && it >= -1e-6f && it <= 1.000001f } && abs(scores.sum() - 1f) < 1e-4f)
        return scores
    }
    @Synchronized
    override fun close() { interpreter.close() }

    private fun inspectQuality(samples: FloatArray, sampleRate: Int): AudioQuality {
        val errors = mutableListOf<String>(); val warnings = mutableListOf<String>()
        if (sampleRate != 8000) errors += "Expected actual 8000Hz audio; resample correctly before inference"
        if (samples.size < 40000) errors += "Record at least five seconds"
        if (samples.any { !it.isFinite() || it < -1f || it > 1f }) errors += "Invalid PCM samples"
        var sumSquares = 0.0; var clipped = 0
        if (errors.isEmpty()) samples.forEach { value ->
            sumSquares += value.toDouble() * value; if (abs(value) >= .999f) clipped++
        }
        val rms = if (samples.isNotEmpty() && errors.isEmpty()) sqrt(sumSquares / samples.size) else 0.0
        val fraction = if (samples.isNotEmpty()) clipped.toDouble() / samples.size else 0.0
        if (errors.isEmpty() && rms < 1e-5) errors += "Recording is silent"
        else if (errors.isEmpty() && rms < 5e-4) warnings += "Recording level is weak"
        if (fraction >= .001) warnings += "Recording may be clipped"
        return AudioQuality(errors.isEmpty(), errors, warnings,
            if (sampleRate > 0) samples.size.toDouble() / sampleRate else 0.0, rms, fraction)
    }
}
