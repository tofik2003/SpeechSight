package com.speechsight.ai.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.speechsight.ai.ui.theme.PrimaryBlue
import com.speechsight.ai.ui.theme.AccentPurple

@Composable
fun SpeakerPill(
    speakerLabel: String,
    isActive: Boolean = true,
    modifier: Modifier = Modifier
) {
    val bgBrush = if (isActive) {
        Brush.horizontalGradient(listOf(PrimaryBlue, AccentPurple))
    } else {
        Brush.horizontalGradient(listOf(Color(0xFF374151), Color(0xFF1F2937)))
    }

    Text(
        text = if (isActive) "● $speakerLabel (Active)" else speakerLabel,
        color = Color.White,
        fontSize = 12.sp,
        fontWeight = FontWeight.Bold,
        modifier = modifier
            .background(brush = bgBrush, shape = RoundedCornerShape(12.dp))
            .padding(horizontal = 10.dp, vertical = 4.dp)
    )
}
