package com.ausculta.model

import java.nio.ByteBuffer
import java.nio.ByteOrder

data class DecodedPcm16(val samples: ShortArray, val sampleRate: Int)

/** Strict RIFF/WAVE decoder for mono 8kHz signed16-bit PCM; skips metadata chunks. */
object WavPcm16 {
    fun decode(bytes: ByteArray): DecodedPcm16 {
        require(bytes.size >= 12) { "Truncated WAV header" }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(buffer.getInt(0) == 0x46464952 && buffer.getInt(8) == 0x45564157) { "Expected RIFF/WAVE file" }
        val riffEnd = (buffer.getInt(4).toLong() and 0xffffffffL) + 8
        require(riffEnd >= 12 && riffEnd <= bytes.size) { "Truncated RIFF data" }
        var position = 12L; var formatSeen = false; var dataStart = -1; var dataSize = 0
        while (position + 8 <= riffEnd) {
            val p = position.toInt(); val id = buffer.getInt(p)
            val size = buffer.getInt(p + 4).toLong() and 0xffffffffL
            val start = position + 8; val end = start + size
            require(end <= riffEnd) { "Truncated WAV chunk" }
            if (id == 0x20746d66) {
                require(!formatSeen && size >= 16) { "Invalid fmt chunk" }; formatSeen = true
                val offset = start.toInt()
                require(buffer.getShort(offset).toInt() == 1) { "Only uncompressed PCM WAV is supported" }
                require(buffer.getShort(offset + 2).toInt() == 1) { "WAV must be mono" }
                require(buffer.getInt(offset + 4) == 8000) { "WAV must actually be 8000Hz; resample upstream" }
                require(buffer.getInt(offset + 8) == 16000 && buffer.getShort(offset + 12).toInt() == 2 && buffer.getShort(offset + 14).toInt() == 16) { "Expected signed16-bit PCM" }
            } else if (id == 0x61746164) {
                require(dataStart < 0 && size > 0 && size % 2 == 0L) { "Invalid or duplicate data chunk" }
                dataStart = start.toInt(); dataSize = size.toInt()
            }
            position = end + (size % 2)
            require(position <= riffEnd) { "Missing chunk padding" }
        }
        require(position == riffEnd && formatSeen && dataStart >= 0) { "Missing or malformed WAV chunks" }
        val samples = ShortArray(dataSize / 2) { buffer.getShort(dataStart + it * 2) }
        return DecodedPcm16(samples, 8000)
    }
}
