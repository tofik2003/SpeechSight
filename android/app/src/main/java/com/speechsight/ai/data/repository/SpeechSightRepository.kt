package com.speechsight.ai.data.repository

import android.content.Context
import android.net.Uri
import com.speechsight.ai.data.local.*
import com.speechsight.ai.domain.*
import com.speechsight.ai.engine.OnDeviceVideoProcessor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.withContext
import java.util.UUID

/**
 * SpeechSight Local Repository for Android.
 * Orchestrates on-device video processing and Room local database caching.
 */
class SpeechSightRepository(
    private val context: Context,
    private val dao: TranscriptDao
) {
    private val onDeviceProcessor = OnDeviceVideoProcessor(context)

    fun getRecentProjects(): Flow<List<VideoProjectEntity>> = dao.getAllProjects()

    suspend fun processVideoLocally(
        videoUri: Uri,
        mode: ProcessingMode,
        language: String,
        privacyMode: Boolean,
        onProgress: ((PipelineStep) -> Unit)? = null
    ): TranscriptResult = withContext(Dispatchers.Default) {
        val result = onDeviceProcessor.processVideoLocally(
            videoUri = videoUri,
            mode = mode,
            language = language,
            privacyMode = privacyMode,
            onStepProgress = onProgress
        )

        saveCompletedTranscript(
            videoUri = videoUri.toString(),
            result = result,
            title = "On-Device Transcription"
        )

        result
    }

    suspend fun saveCompletedTranscript(
        videoUri: String,
        result: TranscriptResult,
        title: String = "Video Transcription"
    ) = withContext(Dispatchers.IO) {
        val projectId = UUID.randomUUID().toString()
        val duration = result.segments.lastOrNull()?.endTime ?: 0.0f

        val project = VideoProjectEntity(
            id = projectId,
            title = title,
            localUri = videoUri,
            durationSec = duration,
            processingMode = result.processingMode.value,
            language = result.language
        )
        dao.insertProject(project)

        val transcriptId = UUID.randomUUID().toString()
        val transcript = TranscriptEntity(
            id = transcriptId,
            projectId = projectId,
            fullText = result.fullText ?: result.segments.joinToString(" ") { it.text },
            averageConfidence = result.metrics?.averageCombinedConfidence ?: 0.85f,
            processingMode = result.processingMode.value,
            privacyStatus = result.privacyStatus
        )
        dao.insertTranscript(transcript)

        val segmentEntities = result.segments.mapIndexed { index, seg ->
            SegmentEntity(
                id = "${transcriptId}_seg_$index",
                transcriptId = transcriptId,
                startTime = seg.startTime,
                endTime = seg.endTime,
                speaker = seg.speaker,
                text = seg.text,
                audioConfidence = seg.audioConfidence,
                visualConfidence = seg.visualConfidence,
                combinedConfidence = seg.combinedConfidence,
                status = seg.status.value,
                uncertaintyNote = seg.uncertaintyNote
            )
        }
        dao.insertSegments(segmentEntities)
    }

    suspend fun updateSegmentCorrection(segmentId: String, newText: String) = withContext(Dispatchers.IO) {
        dao.updateSegmentText(segmentId, newText)
    }

    suspend fun wipeAllData() = withContext(Dispatchers.IO) {
        dao.wipeAllLocalData()
    }
}
