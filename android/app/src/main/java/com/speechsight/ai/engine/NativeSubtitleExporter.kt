package com.speechsight.ai.engine

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import com.speechsight.ai.domain.TranscriptResult
import com.speechsight.ai.domain.TranscriptSegment
import java.io.File
import java.io.FileWriter
import java.util.Locale

/**
 * On-Device Subtitle Generator and Exporter for Android.
 * Converts transcripts to SubRip (.srt), WebVTT (.vtt), Plain Text (.txt), and JSON formats.
 */
object NativeSubtitleExporter {

    private fun formatTimestampSrt(seconds: Float): String {
        val hrs = (seconds / 3600).toInt()
        val mins = ((seconds % 3600) / 60).toInt()
        val secs = (seconds % 60).toInt()
        val millis = ((seconds - seconds.toInt()) * 1000).toInt()
        return String.format(Locale.US, "%02d:%02d:%02d,%03d", hrs, mins, secs, millis)
    }

    private fun formatTimestampVtt(seconds: Float): String {
        val hrs = (seconds / 3600).toInt()
        val mins = ((seconds % 3600) / 60).toInt()
        val secs = (seconds % 60).toInt()
        val millis = ((seconds - seconds.toInt()) * 1000).toInt()
        return String.format(Locale.US, "%02d:%02d:%02d.%03d", hrs, mins, secs, millis)
    }

    fun toSrt(segments: List<TranscriptSegment>, includeSpeaker: Boolean = true): String {
        val sb = StringBuilder()
        segments.forEachIndexed { idx, seg ->
            val num = idx + 1
            val st = formatTimestampSrt(seg.startTime)
            val et = formatTimestampSrt(seg.endTime)
            val speakerPrefix = if (includeSpeaker && seg.speaker.isNotBlank()) "[${seg.speaker}] " else ""
            sb.append("$num\n")
            sb.append("$st --> $et\n")
            sb.append("$speakerPrefix${seg.text}\n\n")
        }
        return sb.toString().trim()
    }

    fun toVtt(segments: List<TranscriptSegment>, includeSpeaker: Boolean = true): String {
        val sb = StringBuilder()
        sb.append("WEBVTT - Generated On-Device by SpeechSight AI\n\n")
        segments.forEachIndexed { idx, seg ->
            val num = idx + 1
            val st = formatTimestampVtt(seg.startTime)
            val et = formatTimestampVtt(seg.endTime)
            sb.append("$num\n")
            sb.append("$st --> $et\n")
            if (includeSpeaker && seg.speaker.isNotBlank()) {
                sb.append("<v ${seg.speaker}>${seg.text}\n\n")
            } else {
                sb.append("${seg.text}\n\n")
            }
        }
        return sb.toString().trim()
    }

    fun toPlainText(segments: List<TranscriptSegment>): String {
        val sb = StringBuilder()
        segments.forEach { seg ->
            val startMin = (seg.startTime / 60).toInt()
            val startSec = (seg.startTime % 60).toInt()
            val endMin = (seg.endTime / 60).toInt()
            val endSec = (seg.endTime % 60).toInt()
            val timeHeader = String.format(Locale.US, "[%02d:%02d - %02d:%02d]", startMin, startSec, endMin, endSec)
            sb.append("$timeHeader ${seg.speaker}: ${seg.text}\n")
        }
        return sb.toString().trim()
    }

    fun exportToFile(context: Context, format: String, segments: List<TranscriptSegment>): File {
        val content = when (format.lowercase(Locale.ROOT)) {
            "vtt", "webvtt" -> toVtt(segments)
            "txt", "text" -> toPlainText(segments)
            else -> toSrt(segments)
        }
        val ext = when (format.lowercase(Locale.ROOT)) {
            "vtt", "webvtt" -> "vtt"
            "txt", "text" -> "txt"
            else -> "srt"
        }

        val exportDir = File(context.cacheDir, "exports").apply { mkdirs() }
        val file = File(exportDir, "speechsight_subtitles_${System.currentTimeMillis()}.$ext")
        FileWriter(file).use { it.write(content) }
        return file
    }
}
