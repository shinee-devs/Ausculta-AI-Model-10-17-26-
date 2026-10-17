package com.ausculta.model
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.*
import org.junit.Test

class WavPcm16Test {
    private fun wav(withJunk: Boolean = false): ByteArray {
        val size = if (withJunk) 58 else 48
        val b = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN)
        b.putInt(0x46464952).putInt(size - 8).putInt(0x45564157)
        b.putInt(0x20746d66).putInt(16).putShort(1).putShort(1).putInt(8000).putInt(16000).putShort(2).putShort(16)
        if (withJunk) b.putInt(0x4b4e554a).putInt(1).put(7).put(0)
        b.putInt(0x61746164).putInt(4).putShort(-32768).putShort(32767)
        return b.array()
    }
    @Test fun decodeSignedSamples() {
        val decoded = WavPcm16.decode(wav())
        assertEquals(8000, decoded.sampleRate)
        assertArrayEquals(shortArrayOf(-32768, 32767), decoded.samples)
    }
    @Test fun skipOddMetadataChunk() {
        assertArrayEquals(shortArrayOf(-32768, 32767), WavPcm16.decode(wav(true)).samples)
    }
    @Test(expected = IllegalArgumentException::class) fun rejectTruncatedFile() {
        WavPcm16.decode(wav().copyOf(45))
    }
    @Test(expected = IllegalArgumentException::class) fun rejectStereo() {
        val bytes = wav(); ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putShort(22, 2)
        WavPcm16.decode(bytes)
    }
}
