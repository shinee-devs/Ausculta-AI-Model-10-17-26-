package com.ausculta.model
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import java.nio.ByteOrder

@RunWith(AndroidJUnit4::class)
class GoldenInferenceTest {
    @Test fun syntheticFixtureMatchesPortableRuntime() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val bytes = context.assets.open("ausculta/golden_input.f32").use { it.readBytes() }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val windows = Array(3) { FloatArray(40000) { buffer.float } }
        val expected = JSONObject(context.assets.open("ausculta/golden_output.json").bufferedReader().use { it.readText() }).getJSONArray("class_scores")
        AuscultaModel(context).use { model ->
            val actual = model.runWindows(windows)
            for (i in 0..3) assertEquals(expected.getDouble(i), actual[i].toDouble(), 1e-4)
            val silent = model.analyze(FloatArray(40000), 8000)
            assertFalse(silent.quality.valid); assertNull(silent.analysisResult); assertNull(silent.modelScore)
            assertFalse(silent.acceptancePassed); assertFalse(silent.confidenceCalibrated)
        }
    }
}
