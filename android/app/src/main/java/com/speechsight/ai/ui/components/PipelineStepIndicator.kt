package com.speechsight.ai.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.speechsight.ai.ui.theme.*

@Composable
fun PipelineStepIndicator(
    stepNumber: Int,
    stepName: String,
    durationMs: Float,
    isCompleted: Boolean,
    isActive: Boolean,
    details: String? = null,
    modifier: Modifier = Modifier
) {
    val borderColor = when {
        isActive -> PrimaryBlue
        isCompleted -> SuccessGreen
        else -> BorderColor
    }

    val bgColor = when {
        isActive -> Color(0x264F80FF)
        isCompleted -> Color(0x1A10B981)
        else -> DarkSurfaceVariant
    }

    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(bgColor)
            .border(1.dp, borderColor, RoundedCornerShape(8.dp))
            .padding(horizontal = 12.dp, vertical = 8.dp)
    ) {
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .size(24.dp)
                .background(if (isCompleted) SuccessGreen else PrimaryBlue, CircleShape)
        ) {
            Text(
                text = "$stepNumber",
                color = Color.White,
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold
            )
        }

        Spacer(modifier = Modifier.width(12.dp))

        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = stepName,
                color = TextPrimary,
                fontSize = 13.sp,
                fontWeight = FontWeight.SemiBold
            )
            if (details != null && details.isNotBlank()) {
                Text(
                    text = details,
                    color = AccentCyan,
                    fontSize = 10.sp
                )
            }
        }

        if (isCompleted && durationMs > 0) {
            Text(
                text = "${durationMs.toInt()} ms",
                color = SuccessGreen,
                fontSize = 11.sp,
                fontWeight = FontWeight.SemiBold
            )
        }
    }
}
