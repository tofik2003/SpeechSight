package com.speechsight.ai.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.speechsight.ai.ui.theme.*

@Composable
fun SettingsScreen(
    onNavigateBack: () -> Unit,
    onWipeAllLocalData: () -> Unit
) {
    var hardwareAccelerationEnabled by remember { mutableStateOf(true) }
    var zeroRetentionPurgeEnabled by remember { mutableStateOf(true) }

    Scaffold(
        containerColor = DarkBackground,
        topBar = {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                TextButton(onClick = onNavigateBack) {
                    Text("← Back", color = PrimaryBlue)
                }
                Text(
                    text = "On-Device AI Engine & Privacy",
                    style = MaterialTheme.typography.titleMedium
                )
                Spacer(modifier = Modifier.width(48.dp))
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
                Text("Autonomous On-Device AI Architecture", color = PrimaryBlue, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                ListItem(
                    headlineContent = { Text("Local Neural AVSR Engine", color = TextPrimary) },
                    supportingContent = { Text("100% Offline execution on device. No external servers or cloud calls.", color = TextSecondary) },
                    trailingContent = {
                        Text("Active", color = SuccessGreen, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                    },
                    colors = ListItemDefaults.colors(containerColor = DarkSurfaceVariant)
                )
            }

            item {
                ListItem(
                    headlineContent = { Text("INT8 Quantized Model (59.3 KB)", color = TextPrimary) },
                    supportingContent = { Text("Spatiotemporal Lip Encoder + Cross-Modal Fusion + CTC Head", color = TextSecondary) },
                    trailingContent = {
                        Text("Loaded", color = AccentCyan, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                    },
                    colors = ListItemDefaults.colors(containerColor = DarkSurfaceVariant)
                )
            }

            item {
                ListItem(
                    headlineContent = { Text("Android NNAPI / DSP Acceleration", color = TextPrimary) },
                    supportingContent = { Text("Accelerates matrix multiplications using on-device NPU/GPU", color = TextSecondary) },
                    trailingContent = {
                        Switch(
                            checked = hardwareAccelerationEnabled,
                            onCheckedChange = { hardwareAccelerationEnabled = it }
                        )
                    },
                    colors = ListItemDefaults.colors(containerColor = DarkSurfaceVariant)
                )
            }

            item {
                Text("Zero-Retention Privacy Safeguards", color = SuccessGreen, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                ListItem(
                    headlineContent = { Text("Instant Ephemeral Purge", color = TextPrimary) },
                    supportingContent = { Text("Immediately recycles and shreds video frames & mouth crops in RAM", color = TextSecondary) },
                    trailingContent = {
                        Switch(
                            checked = zeroRetentionPurgeEnabled,
                            onCheckedChange = { zeroRetentionPurgeEnabled = it }
                        )
                    },
                    colors = ListItemDefaults.colors(containerColor = DarkSurfaceVariant)
                )
            }

            item {
                Spacer(modifier = Modifier.height(16.dp))
                Button(
                    onClick = onWipeAllLocalData,
                    colors = ButtonDefaults.buttonColors(containerColor = ErrorRed),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Purge All Local Transcripts and Media")
                }
            }
        }
    }
}
