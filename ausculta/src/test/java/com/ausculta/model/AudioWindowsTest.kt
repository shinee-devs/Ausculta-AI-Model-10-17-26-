package com.ausculta.model
import org.junit.Assert.*
import org.junit.Test

class AudioWindowsTest {
    @Test fun firstMiddleLastPreserveOriginalSamples() {
        val samples = FloatArray(80000) { it / 80000f }
        val windows = AudioWindows.fromRecording(samples, 8000)
        assertEquals(samples[0], windows[0][0], 0f)
        assertEquals(samples[20000], windows[1][0], 0f)
        assertEquals(samples[40000], windows[2][0], 0f)
        assertEquals(samples[79999], windows[2][39999], 0f)
    }
    @Test(expected = IllegalArgumentException::class) fun rejectWrongSampleRate() {
        AudioWindows.fromRecording(FloatArray(40000), 44100)
    }
    @Test(expected = IllegalArgumentException::class) fun rejectShortRecording() {
        AudioWindows.fromRecording(FloatArray(39999), 8000)
    }
    @Test fun signedPcmNormalization() {
        val result = AudioWindows.normalizePcm16(shortArrayOf(-32768, 0, 32767))
        assertEquals(-1f, result[0], 0f); assertEquals(0f, result[1], 0f)
        assertEquals(32767f / 32768f, result[2], 0f)
    }
}
