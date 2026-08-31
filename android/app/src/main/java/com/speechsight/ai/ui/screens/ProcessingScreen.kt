package com.speechsight.ai.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.speechsight.ai.domain.PipelineStep
import com.speechsight.ai.ui.components.PipelineStepIndicator
import com.speechsight.ai.ui.theme.*

@Composable
fun ProcessingScreen(
    currentStep: Int,
    steps: List<PipelineStep>
) {
    Scaffold(
        containerColor = DarkBackground
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.height(16.dp))

            CircularProgressIndicator(
                color = PrimaryBlue,
                modifier = Modifier.size(50.dp),
                strokeWidth = 4.dp
            )

            Spacer(modifier = Modifier.height(16.dp))

            Text(
                text = "On-Device Neural AVSR Pipeline",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                color = TextPrimary
            )

            Text(
                text = "Executing Step $currentStep of 10 Locally",
                color = AccentCyan,
                fontWeight = FontWeight.SemiBold,
                fontSize = 13.sp
            )

            Spacer(modifier = Modifier.height(20.dp))

            LazyColumn(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(steps) { step ->
                    PipelineStepIndicator(
                        stepNumber = step.stepNumber,
                        stepName = step.stepName,
                        durationMs = step.durationMs,
                        isCompleted = step.stepNumber < currentStep || step.status == "completed",
                        isActive = step.stepNumber == currentStep,
                        details = step.details
                    )
                }
            }
        }
    }
}
