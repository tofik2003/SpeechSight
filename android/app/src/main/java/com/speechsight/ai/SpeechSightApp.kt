package com.speechsight.ai

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.os.Build
import com.speechsight.ai.data.local.AppDatabase

class SpeechSightApp : Application() {

    lateinit var database: AppDatabase
        private set

    override fun onCreate() {
        super.onCreate()
        instance = this
        database = AppDatabase.getInstance(this)
        createNotificationChannel()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "SpeechSight Processing",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Background audio-visual transcription jobs"
            }
            val manager = getSystemService(NotificationManager::class.java)
            manager?.createNotificationChannel(channel)
        }
    }

    companion object {
        const val CHANNEL_ID = "speechsight_processing_channel"
        lateinit var instance: SpeechSightApp
            private set
    }
}
