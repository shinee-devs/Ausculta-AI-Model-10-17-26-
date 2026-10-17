package com.ausculta.model

/** Three original 5-second views. Input is already 8kHz mono normalized PCM. */
object AudioWindows {
    const val SAMPLE_RATE = 8000
    const val WINDOW_SAMPLES = 40000
    fun fromRecording(samples: FloatArray, sampleRate: Int): Array<FloatArray> {
        require(sampleRate == SAMPLE_RATE) { "Expected actual 8000Hz PCM; do not just change the sample-rate label" }
        require(samples.size >= WINDOW_SAMPLES) { "Record at least five seconds" }
        require(samples.all { it.isFinite() && it >= -1f && it <= 1f }) { "PCM must be finite and normalized to [-1,1]" }
        val maximum = samples.size - WINDOW_SAMPLES
        return intArrayOf(0, maximum / 2, maximum).map { start ->
            samples.copyOfRange(start, start + WINDOW_SAMPLES)
        }.toTypedArray()
    }
    fun normalizePcm16(samples: ShortArray): FloatArray =
        FloatArray(samples.size) { samples[it].toFloat() / 32768f }
}
