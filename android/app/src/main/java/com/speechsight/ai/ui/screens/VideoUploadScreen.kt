package com.speechsight.ai.ui.screens

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
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
import com.speechsight.ai.domain.ProcessingMode
import com.speechsight.ai.ui.theme.*

@Composable
fun VideoUploadScreen(
    onStartAnalysis: (Uri, ProcessingMode, String, Boolean) -> Unit,
    onNavigateSettings: () -> Unit
) {
    var selectedVideoUri by remember { mutableStateOf<Uri?>(null) }
    var selectedMode by remember { mutableStateOf(ProcessingMode.AUDIO_VISUAL) }
    var selectedLanguage by remember { mutableStateOf("en") }
    var privacyModeEnabled by remember { mutableStateOf(true) }

    val videoPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        selectedVideoUri = uri
    }

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
                Column {
                    Text(
                        text = "SpeechSight AI",
                        style = MaterialTheme.typography.headlineMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "100% On-Device Visual Intelligence for Human Speech",
                        color = TextSecondary,
                        fontSize = 12.sp
                    )
                }
                TextButton(onClick = onNavigateSettings) {
                    Text("Settings", color = PrimaryBlue)
                }
            }
        }
    ) { innerPadding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            item {
                // On-Device Badge
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(Color(0x1A10B981))
                        .border(1.dp, Color(0x4D10B981), RoundedCornerShape(8.dp))
                        .padding(horizontal = 12.dp, vertical = 8.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(text = "⚡", fontSize = 16.sp)
                        Spacer(modifier = Modifier.width(8.dp))
                        Column {
                            Text(
                                text = "Autonomous On-Device Execution",
                                color = SuccessGreen,
                                fontWeight = FontWeight.Bold,
                                fontSize = 12.sp
                            )
                            Text(
                                text = "Zero server calls • 100% private & offline • Hardware accelerated",
                                color = TextSecondary,
                                fontSize = 11.sp
                            )
                        }
                    }
                }
            }

            item {
                Text(
                    text = "1. Processing Mode",
                    style = MaterialTheme.typography.titleMedium,
                    color = TextPrimary
                )
                Spacer(modifier = Modifier.height(8.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    ModeCard(
                        title = "Audio + Visual",
                        subtitle = "Dynamic Fusion (Recommended)",
                        icon = "🎙️👄",
                        isSelected = selectedMode == ProcessingMode.AUDIO_VISUAL,
                        onClick = { selectedMode = ProcessingMode.AUDIO_VISUAL },
                        modifier = Modifier.weight(1f)
                    )
                    ModeCard(
                        title = "Silent Reading",
                        subtitle = "Visual Lip Only",
                        icon = "🔇👄",
                        isSelected = selectedMode == ProcessingMode.SILENT_VISUAL,
                        onClick = { selectedMode = ProcessingMode.SILENT_VISUAL },
                        modifier = Modifier.weight(1f)
                    )
                }
            }

            item {
                Text(
                    text = "2. Select Video to Process",
                    style = MaterialTheme.typography.titleMedium,
                    color = TextPrimary
                )
                Spacer(modifier = Modifier.height(8.dp))

                Box(
                    contentAlignment = Alignment.Center,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(130.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(DarkSurfaceVariant)
                        .border(1.dp, BorderColor, RoundedCornerShape(12.dp))
                        .clickable { videoPickerLauncher.launch("video/*") }
                        .padding(16.dp)
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = if (selectedVideoUri != null) "✓ Video Selected" else "📁 Choose Local Video File",
                            color = if (selectedVideoUri != null) SuccessGreen else PrimaryBlue,
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = selectedVideoUri?.lastPathSegment ?: "Supports MP4, MOV, MKV, WebM",
                            color = TextSecondary,
                            fontSize = 12.sp
                        )
                    }
                }
            }

            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text("Zero-Retention Privacy", color = TextPrimary, fontWeight = FontWeight.SemiBold)
                        Text("Auto-purge temporary video frames & mouth crops in RAM", color = TextSecondary, fontSize = 11.sp)
                    }
                    Switch(
                        checked = privacyModeEnabled,
                        onCheckedChange = { privacyModeEnabled = it }
                    )
                }
            }

            item {
                Button(
                    onClick = {
                        val uri = selectedVideoUri ?: Uri.parse("asset:///keynote_presentation.mp4")
                        onStartAnalysis(uri, selectedMode, selectedLanguage, privacyModeEnabled)
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(52.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = PrimaryBlue),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Text(
                        text = "Run On-Device AVSR Analysis",
                        fontWeight = FontWeight.Bold,
                        fontSize = 16.sp
                    )
                }
            }
        }
    }
}

@Composable
fun ModeCard(
    title: String,
    subtitle: String,
    icon: String,
    isSelected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(10.dp))
            .background(if (isSelected) Color(0x264F80FF) else DarkSurfaceVariant)
            .border(1.dp, if (isSelected) PrimaryBlue else BorderColor, RoundedCornerShape(10.dp))
            .clickable(onClick = onClick)
            .padding(12.dp)
    ) {
        Column {
            Text(text = icon, fontSize = 20.sp)
            Spacer(modifier = Modifier.height(4.dp))
            Text(text = title, color = TextPrimary, fontWeight = FontWeight.Bold, fontSize = 13.sp)
            Text(text = subtitle, color = TextSecondary, fontSize = 10.sp)
        }
    }
}
