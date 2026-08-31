package com.speechsight.ai.ui.viewmodel

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.speechsight.ai.SpeechSightApp
import com.speechsight.ai.data.repository.SpeechSightRepository
import com.speechsight.ai.domain.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed class UiState {
    object Idle : UiState()
    data class Processing(val currentStep: Int, val steps: List<PipelineStep>) : UiState()
    data class Success(val videoUri: Uri, val result: TranscriptResult) : UiState()
    data class Error(val message: String) : UiState()
}

class SpeechSightViewModel(application: Application) : AndroidViewModel(application) {

    private val repository = SpeechSightRepository(
        context = application.applicationContext,
        dao = (application as SpeechSightApp).database.transcriptDao()
    )

    private val _uiState = MutableStateFlow<UiState>(UiState.Idle)
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    fun processVideo(
        videoUri: Uri,
        mode: ProcessingMode,
        language: String,
        privacyMode: Boolean
    ) {
        viewModelScope.launch {
            val initialSteps = mutableListOf(
                PipelineStep(1, "Video decoding & frame separation", 0f, "pending"),
                PipelineStep(2, "Face detection & landmark alignment", 0f, "pending"),
                PipelineStep(3, "Face tracking & identity persistence", 0f, "pending"),
                PipelineStep(4, "Mouth ROI crop extraction", 0f, "pending"),
                PipelineStep(5, "Active speaker detection & synchrony", 0f, "pending"),
                PipelineStep(6, "Audio acoustic feature analysis", 0f, "pending"),
                PipelineStep(7, "Visual lip-reading model inference", 0f, "pending"),
                PipelineStep(8, "Multimodal prediction fusion", 0f, "pending"),
                PipelineStep(9, "Language correction & uncertainty", 0f, "pending"),
                PipelineStep(10, "Subtitle generation & timestamping", 0f, "pending")
            )
            _uiState.value = UiState.Processing(1, initialSteps)

            try {
                val result = repository.processVideoLocally(
                    videoUri = videoUri,
                    mode = mode,
                    language = language,
                    privacyMode = privacyMode,
                    onProgress = { completedStep ->
                        val updated = initialSteps.map { s ->
                            when {
                                s.stepNumber < completedStep.stepNumber -> s.copy(status = "completed")
                                s.stepNumber == completedStep.stepNumber -> completedStep
                                s.stepNumber == completedStep.stepNumber + 1 -> s.copy(status = "running")
                                else -> s
                            }
                        }
                        _uiState.value = UiState.Processing(completedStep.stepNumber, updated)
                    }
                )
                _uiState.value = UiState.Success(videoUri, result)
            } catch (e: Exception) {
                _uiState.value = UiState.Error(e.message ?: "On-Device processing encountered an error")
            }
        }
    }

    fun resetState() {
        _uiState.value = UiState.Idle
    }

    fun wipeAllData() {
        viewModelScope.launch {
            repository.wipeAllData()
            _uiState.value = UiState.Idle
        }
    }
}
