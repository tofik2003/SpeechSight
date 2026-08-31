package com.speechsight.ai.domain

enum class ProcessingMode(val value: String) {
    AUDIO_VISUAL("audio_visual"),
    SILENT_VISUAL("silent_visual"),
    AUDIO_ONLY("audio_only")
}

enum class ConfidenceStatus(val value: String) {
    HIGH_CONFIDENCE("high_confidence"),
    MEDIUM_CONFIDENCE("medium_confidence"),
    LOW_CONFIDENCE("low_confidence"),
    UNCERTAIN_VISUAL_ONLY("uncertain_visual_only")
}

data class BoundingBox(
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float
)

data class FaceLandmarks(
    val mouthLeft: FloatArray? = null,
    val mouthRight: FloatArray? = null,
    val mouthTop: FloatArray? = null,
    val mouthBottom: FloatArray? = null,
    val lipOpening: Float = 0.15f
)

data class FaceTrack(
    val trackId: Int,
    val speakerLabel: String,
    val frameIndex: Int,
    val timestamp: Float,
    val bbox: BoundingBox,
    val landmarks: FaceLandmarks? = null,
    var isActiveSpeaker: Boolean = false,
    var speakerScore: Float = 0.0f
)

data class WordToken(
    val word: String,
    val startTime: Float,
    val endTime: Float,
    val confidence: Float,
    val audioConfidence: Float? = null,
    val visualConfidence: Float? = null
)

data class TranscriptSegment(
    val id: String = "",
    val startTime: Float,
    val endTime: Float,
    val speaker: String = "Speaker 1",
    val text: String,
    val audioConfidence: Float = 0.0f,
    val visualConfidence: Float = 0.0f,
    val combinedConfidence: Float = 0.0f,
    val status: ConfidenceStatus = ConfidenceStatus.HIGH_CONFIDENCE,
    val words: List<WordToken> = emptyList(),
    val uncertaintyNote: String? = null
)

data class PipelineStep(
    val stepNumber: Int,
    val stepName: String,
    val durationMs: Float,
    val status: String,
    val details: String? = null
)

data class PipelineMetrics(
    val totalDurationSec: Float,
    val videoDurationSec: Float,
    val realTimeFactor: Float,
    val activeSpeakersDetected: Int,
    val averageCombinedConfidence: Float,
    val steps: List<PipelineStep> = emptyList()
)

data class TranscriptResult(
    val videoId: String,
    val language: String = "en",
    val segments: List<TranscriptSegment>,
    val processingMode: ProcessingMode = ProcessingMode.AUDIO_VISUAL,
    val privacyStatus: String = "temporary_media_deleted",
    val metrics: PipelineMetrics? = null,
    val fullText: String? = null
)
