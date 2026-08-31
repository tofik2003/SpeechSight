package com.speechsight.ai.data.local

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface TranscriptDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertProject(project: VideoProjectEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertTranscript(transcript: TranscriptEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertSegments(segments: List<SegmentEntity>)

    @Query("SELECT * FROM video_projects ORDER BY createdAt DESC")
    fun getAllProjects(): Flow<List<VideoProjectEntity>>

    @Query("SELECT * FROM transcripts WHERE projectId = :projectId LIMIT 1")
    suspend fun getTranscriptForProject(projectId: String): TranscriptEntity?

    @Query("SELECT * FROM segments WHERE transcriptId = :transcriptId ORDER BY startTime ASC")
    fun getSegmentsForTranscript(transcriptId: String): Flow<List<SegmentEntity>>

    @Query("UPDATE segments SET text = :newText WHERE id = :segmentId")
    suspend fun updateSegmentText(segmentId: String, newText: String)

    @Query("DELETE FROM video_projects WHERE id = :projectId")
    suspend fun deleteProject(projectId: String)

    @Query("DELETE FROM video_projects")
    suspend fun wipeAllLocalData()
}
