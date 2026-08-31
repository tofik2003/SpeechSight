package com.speechsight.ai

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.navigation.compose.rememberNavController
import com.speechsight.ai.ui.navigation.AppNavHost
import com.speechsight.ai.ui.theme.DarkBackground
import com.speechsight.ai.ui.theme.SpeechSightTheme
import com.speechsight.ai.ui.viewmodel.SpeechSightViewModel

class MainActivity : ComponentActivity() {

    private val viewModel: SpeechSightViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            SpeechSightTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = DarkBackground
                ) {
                    val navController = rememberNavController()
                    AppNavHost(
                        navController = navController,
                        viewModel = viewModel
                    )
                }
            }
        }
    }
}
