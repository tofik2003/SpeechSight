package com.speechsight.ai.engine

import com.speechsight.ai.domain.ConfidenceStatus
import com.speechsight.ai.domain.ProcessingMode
import com.speechsight.ai.domain.TranscriptSegment
import com.speechsight.ai.domain.WordToken
import kotlin.math.max
import kotlin.math.min

/**
 * On-Device Dynamic Audio-Visual Fusion Engine for Android.
 * Adapts fusion weights based on acoustic SNR and visual lip clarity.
 */
class NativeFusionEngine {

    fun fusePredictions(
        startTime: Float,
        endTime: Float,
        speaker: String,
        hypothesisText: String,
        mode: ProcessingMode,
        rawVisualConf: Float,
        rawAudioConf: Float,
        snrDb: Float = 20.0f
    ): TranscriptSegment {
        val visualConf = rawVisualConf.coerceIn(0.20f, 0.95f)
        val audioConf = if (mode == ProcessingMode.SILENT_VISUAL) 0.05f else rawAudioConf.coerceIn(0.30f, 0.98f)

        val (combinedConf, status, uncertaintyNote) = when (mode) {
            ProcessingMode.SILENT_VISUAL -> {
                val conf = visualConf
                val stat = if (conf < 0.70f) ConfidenceStatus.UNCERTAIN_VISUAL_ONLY else ConfidenceStatus.MEDIUM_CONFIDENCE
                val note = "Visual-only lip-reading prediction (muted/silent audio). Verify high-stakes words."
                Triple(conf, stat, note)
            }
            ProcessingMode.AUDIO_ONLY -> {
                val conf = audioConf
                val stat = if (conf >= 0.80f) ConfidenceStatus.HIGH_CONFIDENCE else ConfidenceStatus.MEDIUM_CONFIDENCE
                Triple(conf, stat, null)
            }
            ProcessingMode.AUDIO_VISUAL -> {
                val (wAudio, wVisual) = if (snrDb < 10.0f) {
                    Pair(0.35f, 0.65f) // Noisy environment -> rely heavily on visual lip movement
                } else {
                    Pair(0.70f, 0.30f) // Clear environment -> synergize audio + visual
                }

                var combined = (audioConf * wAudio) + (visualConf * wVisual)
                if (audioConf > 0.60f && visualConf > 0.60f) {
                    combined = min(0.99f, combined * 1.05f) // Multimodal concordance bonus
                }

                val stat = if (combined >= 0.80f) {
                    ConfidenceStatus.HIGH_CONFIDENCE
                } else if (combined >= 0.55f) {
                    ConfidenceStatus.MEDIUM_CONFIDENCE
                } else {
                    ConfidenceStatus.LOW_CONFIDENCE
                }

                val note = if (stat == ConfidenceStatus.MEDIUM_CONFIDENCE) {
                    "Moderate confidence; visual lip movements assisted acoustic decoding."
                } else null

                Triple(combined, stat, note)
            }
        }

        // Generate timestamped word tokens
        val rawWords = hypothesisText.trim().split("\\s+".toRegex()).filter { it.isNotBlank() }
        val wordTokens = mutableListOf<WordToken>()
        val count = max(1, rawWords.size)
        val duration = max(0.1f, endTime - startTime)
        val step = duration / count

        for (i in rawWords.indices) {
            val wStart = startTime + i * step
            val wEnd = startTime + (i + 1) * step
            wordTokens.add(
                WordToken(
                    word = rawWords[i],
                    startTime = wStart,
                    endTime = wEnd,
                    confidence = combinedConf,
                    audioConfidence = audioConf,
                    visualConfidence = visualConf
                )
            )
        }

        return TranscriptSegment(
            id = "seg_${System.currentTimeMillis()}_${(0..999).random()}",
            startTime = startTime,
            endTime = endTime,
            speaker = speaker,
            text = hypothesisText.trim(),
            audioConfidence = audioConf,
            visualConfidence = visualConf,
            combinedConfidence = combinedConf,
            status = status,
            words = wordTokens,
            uncertaintyNote = uncertaintyNote
        )
    }
}
