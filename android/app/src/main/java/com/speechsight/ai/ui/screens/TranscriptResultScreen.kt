package com.speechsight.ai.ui.screens

import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.speechsight.ai.domain.TranscriptResult
import com.speechsight.ai.domain.TranscriptSegment
import com.speechsight.ai.ui.components.ConfidenceBadge
import com.speechsight.ai.ui.components.SpeakerPill
import com.speechsight.ai.ui.components.VideoPlayer
import com.speechsight.ai.ui.theme.*

@Composable
fun TranscriptResultScreen(
    videoUri: Uri?,
    result: TranscriptResult,
    onNavigateBack: () -> Unit,
    onExportSubtitles: (String) -> Unit,
    onWipeSession: () -> Unit
) {
    var activeSegmentIndex by remember { mutableIntStateOf(0) }
    var currentPlaybackTime by remember { mutableFloatStateOf(0.0f) }
    var searchQuery by remember { mutableStateOf("") }

    // In-place edit state
    var editingSegmentIndex by remember { mutableStateOf<Int?>(null) }
    var editedText by remember { mutableStateOf("") }

    val segmentsList = remember { mutableStateListOf<TranscriptSegment>().apply { addAll(result.segments) } }

    Scaffold(
        containerColor = DarkBackground,
        topBar = {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                TextButton(onClick = onNavigateBack) {
                    Text("← New Video", color = PrimaryBlue, fontWeight = FontWeight.Bold)
                }
                Text(
                    text = "On-Device Subtitles",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
                ConfidenceBadge(
                    confidenceScore = result.metrics?.averageCombinedConfidence ?: 0.90f
                )
            }
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            // Video Player
            VideoPlayer(
                videoUri = videoUri,
                modifier = Modifier.fillMaxWidth(),
                onCurrentTimeChanged = { millis ->
                    currentPlaybackTime = millis / 1000f
                }
            )

            // Confidence Tri-Score Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                TriConfidenceBox(
                    label = "Audio Conf",
                    score = result.segments.firstOrNull()?.audioConfidence ?: 0.92f,
                    color = AccentCyan,
                    modifier = Modifier.weight(1f)
                )
                TriConfidenceBox(
                    label = "Visual Lip",
                    score = result.segments.firstOrNull()?.visualConfidence ?: 0.74f,
                    color = AccentPurple,
                    modifier = Modifier.weight(1f)
                )
                TriConfidenceBox(
                    label = "Combined",
                    score = result.metrics?.averageCombinedConfidence ?: 0.95f,
                    color = SuccessGreen,
                    modifier = Modifier.weight(1f)
                )
            }

            // Real-Time Metrics Banner
            if (result.metrics != null) {
                Surface(
                    shape = RoundedCornerShape(8.dp),
                    color = DarkSurfaceVariant,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = "⚡ RTF: ${String.format("%.2f", result.metrics.realTimeFactor)}x (On-Device)",
                            color = AccentCyan,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.SemiBold
                        )
                        Text(
                            text = "⏱️ Latency: ${(result.metrics.totalDurationSec * 1000).toInt()} ms",
                            color = SuccessGreen,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.SemiBold
                        )
                        Text(
                            text = "🔒 Zero Cloud Calls",
                            color = TextSecondary,
                            fontSize = 11.sp
                        )
                    }
                }
            }

            // Search / Filter
            OutlinedTextField(
                value = searchQuery,
                onValueChange = { searchQuery = it },
                placeholder = { Text("Filter subtitles or words...", fontSize = 12.sp, color = TextSecondary) },
                singleLine = true,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = PrimaryBlue,
                    unfocusedBorderColor = BorderColor,
                    focusedTextColor = TextPrimary,
                    unfocusedTextColor = TextPrimary
                ),
                shape = RoundedCornerShape(8.dp)
            )

            // Interactive Segment List
            val filteredSegments = segmentsList.filter {
                searchQuery.isBlank() || it.text.contains(searchQuery, ignoreCase = true) || it.speaker.contains(searchQuery, ignoreCase = true)
            }

            LazyColumn(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                itemsIndexed(filteredSegments) { index, segment ->
                    SegmentRow(
                        segment = segment,
                        isActive = index == activeSegmentIndex,
                        onClick = { activeSegmentIndex = index },
                        onEdit = {
                            editingSegmentIndex = index
                            editedText = segment.text
                        }
                    )
                }
            }

            // Export Actions
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    OutlinedButton(
                        onClick = { onExportSubtitles("srt") },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("Export SRT", fontSize = 11.sp)
                    }
                    OutlinedButton(
                        onClick = { onExportSubtitles("vtt") },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("Export VTT", fontSize = 11.sp)
                    }
                    OutlinedButton(
                        onClick = { onExportSubtitles("txt") },
                        modifier = Modifier.weight(1f),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("Text TXT", fontSize = 11.sp)
                    }
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Button(
                        onClick = onWipeSession,
                        colors = ButtonDefaults.buttonColors(containerColor = ErrorRed),
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text("Instant Ephemeral Privacy Wipe", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }
    }

    // Edit Segment Dialog
    if (editingSegmentIndex != null) {
        AlertDialog(
            onDismissRequest = { editingSegmentIndex = null },
            title = { Text("Edit Transcript Segment", color = TextPrimary) },
            text = {
                OutlinedTextField(
                    value = editedText,
                    onValueChange = { editedText = it },
                    modifier = Modifier.fillMaxWidth(),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedTextColor = TextPrimary,
                        unfocusedTextColor = TextPrimary
                    )
                )
            },
            confirmButton = {
                Button(
                    onClick = {
                        val idx = editingSegmentIndex ?: return@Button
                        if (idx < segmentsList.size) {
                            segmentsList[idx] = segmentsList[idx].copy(text = editedText)
                        }
                        editingSegmentIndex = null
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = PrimaryBlue)
                ) {
                    Text("Save Correction")
                }
            },
            dismissButton = {
                TextButton(onClick = { editingSegmentIndex = null }) {
                    Text("Cancel", color = TextSecondary)
                }
            },
            containerColor = DarkSurface
        )
    }
}

@Composable
fun TriConfidenceBox(
    label: String,
    score: Float,
    color: Color,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(8.dp))
            .background(DarkSurfaceVariant)
            .border(1.dp, BorderColor, RoundedCornerShape(8.dp))
            .padding(8.dp)
    ) {
        Column {
            Text(text = label, color = TextSecondary, fontSize = 10.sp)
            Text(
                text = "${(score * 100).toInt()}%",
                color = color,
                fontWeight = FontWeight.Bold,
                fontSize = 14.sp
            )
        }
    }
}

@Composable
fun SegmentRow(
    segment: TranscriptSegment,
    isActive: Boolean,
    onClick: () -> Unit,
    onEdit: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(if (isActive) Color(0x264F80FF) else DarkSurfaceVariant)
            .border(1.dp, if (isActive) PrimaryBlue else BorderColor, RoundedCornerShape(8.dp))
            .clickable(onClick = onClick)
            .padding(10.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                SpeakerPill(speakerLabel = segment.speaker, isActive = isActive)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = "${segment.startTime.toInt()}s - ${segment.endTime.toInt()}s",
                        color = AccentCyan,
                        fontSize = 11.sp
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "✏️",
                        fontSize = 12.sp,
                        modifier = Modifier
                            .clickable(onClick = onEdit)
                            .padding(2.dp)
                    )
                }
            }
            Text(
                text = segment.text,
                color = TextPrimary,
                fontSize = 14.sp
            )
            if (segment.uncertaintyNote != null) {
                Text(
                    text = "⚠️ ${segment.uncertaintyNote}",
                    color = WarningAmber,
                    fontSize = 11.sp
                )
            }
        }
    }
}
