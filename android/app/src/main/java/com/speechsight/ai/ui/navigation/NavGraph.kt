package com.speechsight.ai.ui.navigation

import android.content.Intent
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.FileProvider
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.speechsight.ai.engine.NativeSubtitleExporter
import com.speechsight.ai.ui.screens.*
import com.speechsight.ai.ui.viewmodel.SpeechSightViewModel
import com.speechsight.ai.ui.viewmodel.UiState

sealed class Screen(val route: String) {
    object Upload : Screen("upload")
    object Settings : Screen("settings")
}

@Composable
fun AppNavHost(
    navController: NavHostController,
    viewModel: SpeechSightViewModel
) {
    val uiState by viewModel.uiState.collectAsState()
    val context = LocalContext.current

    NavHost(
        navController = navController,
        startDestination = Screen.Upload.route
    ) {
        composable(Screen.Upload.route) {
            when (val state = uiState) {
                is UiState.Idle -> {
                    VideoUploadScreen(
                        onStartAnalysis = { uri, mode, lang, privacy ->
                            viewModel.processVideo(uri, mode, lang, privacy)
                        },
                        onNavigateSettings = { navController.navigate(Screen.Settings.route) }
                    )
                }
                is UiState.Processing -> {
                    ProcessingScreen(
                        currentStep = state.currentStep,
                        steps = state.steps
                    )
                }
                is UiState.Success -> {
                    TranscriptResultScreen(
                        videoUri = state.videoUri,
                        result = state.result,
                        onNavigateBack = { viewModel.resetState() },
                        onExportSubtitles = { format ->
                            val file = NativeSubtitleExporter.exportToFile(context, format, state.result.segments)
                            val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
                            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                                type = "text/plain"
                                putExtra(Intent.EXTRA_STREAM, uri)
                                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                            }
                            context.startActivity(Intent.createChooser(shareIntent, "Export Subtitles"))
                        },
                        onWipeSession = { viewModel.wipeAllData() }
                    )
                }
                is UiState.Error -> {
                    VideoUploadScreen(
                        onStartAnalysis = { uri, mode, lang, privacy ->
                            viewModel.processVideo(uri, mode, lang, privacy)
                        },
                        onNavigateSettings = { navController.navigate(Screen.Settings.route) }
                    )
                }
            }
        }

        composable(Screen.Settings.route) {
            SettingsScreen(
                onNavigateBack = { navController.popBackStack() },
                onWipeAllLocalData = { viewModel.wipeAllData() }
            )
        }
    }
}
