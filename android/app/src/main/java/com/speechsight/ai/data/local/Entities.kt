package com.speechsight.ai.data.local

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(tableName = "video_projects")
data class VideoProjectEntity(
    @PrimaryKey val id: String,
    val title: String,
    val localUri: String,
    val durationSec: Float,
    val processingMode: String,
    val language: String,
    val createdAt: Long = System.currentTimeMillis()
)

@Entity(
    tableName = "transcripts",
    foreignKeys = [
        ForeignKey(
            entity = VideoProjectEntity::class,
            parentColumns = ["id"],
            childColumns = ["projectId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index("projectId")]
)
data class TranscriptEntity(
    @PrimaryKey val id: String,
    val projectId: String,
    val fullText: String,
    val averageConfidence: Float,
    val processingMode: String,
    val privacyStatus: String,
    val createdAt: Long = System.currentTimeMillis()
)

@Entity(
    tableName = "segments",
    foreignKeys = [
        ForeignKey(
            entity = TranscriptEntity::class,
            parentColumns = ["id"],
            childColumns = ["transcriptId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index("transcriptId")]
)
data class SegmentEntity(
    @PrimaryKey val id: String,
    val transcriptId: String,
    val startTime: Float,
    val endTime: Float,
    val speaker: String,
    val text: String,
    val audioConfidence: Float,
    val visualConfidence: Float,
    val combinedConfidence: Float,
    val status: String,
    val uncertaintyNote: String?
)
