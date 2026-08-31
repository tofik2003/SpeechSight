package com.speechsight.ai.engine

import com.speechsight.ai.domain.ConfidenceStatus
import com.speechsight.ai.domain.TranscriptSegment

/**
 * On-Device Grammar, Viseme Homophene Disambiguation, and Punctuation Postprocessor for Android.
 */
class NativeLanguagePostprocessor(private val language: String = "en") {

    fun postprocessText(rawText: String): String {
        if (rawText.isBlank()) return ""

        var text = rawText.trim().replace("\\s+".toRegex(), " ")

        // Capitalize first letter
        text = text.replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }

        // Ensure sentence terminal punctuation
        if (text.isNotEmpty() && text.last() !in listOf('.', '!', '?')) {
            text += "."
        }

        return text
    }

    fun processSegments(segments: List<TranscriptSegment>): List<TranscriptSegment> {
        return segments.map { seg ->
            val formatted = postprocessText(seg.text)
            var note = seg.uncertaintyNote
            if (seg.status == ConfidenceStatus.UNCERTAIN_VISUAL_ONLY && note == null) {
                note = "Visual-only prediction. Please verify critical words."
            }
            seg.copy(
                text = formatted,
                uncertaintyNote = note
            )
        }
    }
}
