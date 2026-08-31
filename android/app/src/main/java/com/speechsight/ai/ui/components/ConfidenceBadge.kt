package com.speechsight.ai.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.speechsight.ai.ui.theme.SuccessGreen
import com.speechsight.ai.ui.theme.WarningAmber
import com.speechsight.ai.ui.theme.ErrorRed

@Composable
fun ConfidenceBadge(
    confidenceScore: Float,
    modifier: Modifier = Modifier
) {
    val percent = (confidenceScore * 100).toInt()
    val (bgColor, textColor, borderColor) = when {
        percent >= 80 -> Triple(
            Color(0x2610B981),
            SuccessGreen,
            Color(0x4D10B981)
        )
        percent >= 55 -> Triple(
            Color(0x26F59E0B),
            WarningAmber,
            Color(0x4DF59E0B)
        )
        else -> Triple(
            Color(0x26EF4444),
            ErrorRed,
            Color(0x4DEF4444)
        )
    }

    Text(
        text = "$percent% Conf",
        color = textColor,
        fontSize = 11.sp,
        fontWeight = FontWeight.Bold,
        modifier = modifier
            .background(bgColor, RoundedCornerShape(10.dp))
            .border(1.dp, borderColor, RoundedCornerShape(10.dp))
            .padding(horizontal = 8.dp, vertical = 3.dp)
    )
}
