# SpeechSight AI - On-Device Android ProGuard & R8 Optimization Rules

# 1. Moshi JSON Reflection & Serialization
-keepattributes *Annotation*, Signature, InnerClasses, EnclosingMethod
-keep class com.squareup.moshi.** { *; }
-keep interface com.squareup.moshi.** { *; }
-dontwarn com.squareup.moshi.**
-keepclassmembers class * {
    @com.squareup.moshi.Json <fields>;
    @com.squareup.moshi.JsonClass <fields>;
}
-keep class com.speechsight.ai.domain.** { *; }

# 2. Room Local Database
-keep class androidx.room.** { *; }
-keep class * extends androidx.room.RoomDatabase
-dontwarn androidx.room.paging.**
-keep class com.speechsight.ai.data.local.** { *; }

# 3. Android Media3 & ExoPlayer
-keep class androidx.media3.exoplayer.** { *; }
-keep class androidx.media3.ui.** { *; }
-keep class androidx.media3.common.** { *; }
-dontwarn androidx.media3.**

# 4. Kotlin Coroutines
-keepclassmembers class kotlinx.coroutines.** {
    volatile <fields>;
}
-dontwarn kotlinx.coroutines.**

# 5. WorkManager
-keep class * extends androidx.work.Worker { *; }
-keep class * extends androidx.work.ListenableWorker { *; }
-keep class com.speechsight.ai.workers.** { *; }
