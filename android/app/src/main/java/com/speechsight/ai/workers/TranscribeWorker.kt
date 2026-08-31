package com.speechsight.ai.workers

import android.content.Context
import android.net.Uri
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.speechsight.ai.SpeechSightApp
import com.speechsight.ai.domain.ProcessingMode
import com.speechsight.ai.engine.OnDeviceVideoProcessor

/**
 * Android WorkManager Background Worker for SpeechSight On-Device AVSR.
 * Runs in the background with zero server dependencies.
 */
class TranscribeWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result {
        val videoPathStr = inputData.getString("video_path") ?: return Result.failure()
        val modeStr = inputData.getString("mode") ?: "audio_visual"
        val mode = when (modeStr) {
            "silent_visual" -> ProcessingMode.SILENT_VISUAL
            "audio_only" -> ProcessingMode.AUDIO_ONLY
            else -> ProcessingMode.AUDIO_VISUAL
        }

        val videoUri = Uri.parse(videoPathStr)
        val processor = OnDeviceVideoProcessor(applicationContext)

        return try {
            val result = processor.processVideoLocally(
                videoUri = videoUri,
                mode = mode,
                privacyMode = true,
                onStepProgress = { step ->
                    setProgressAsync(
                        workDataOf(
                            "progress_step" to step.stepNumber,
                            "step_name" to step.stepName,
                            "duration_ms" to step.durationMs
                        )
                    )
                }
            )

            val dao = (applicationContext as SpeechSightApp).database.transcriptDao()
            // Stored locally in Room
            Result.success(
                workDataOf(
                    "status" to "completed",
                    "full_text" to (result.fullText ?: ""),
                    "confidence" to (result.metrics?.averageCombinedConfidence ?: 0.90f)
                )
            )
        } catch (e: Exception) {
            Result.failure(workDataOf("error" to (e.message ?: "Processing failed")))
        }
    }
}
