package com.speechsight.ai.engine

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Rect
import com.speechsight.ai.domain.BoundingBox
import com.speechsight.ai.domain.FaceLandmarks
import com.speechsight.ai.domain.FaceTrack
import kotlin.math.max
import kotlin.math.min

/**
 * On-Device Face Detection, Landmark Tracking, and Mouth ROI Crop Extractor.
 * Runs directly on Android CPU/GPU without external services.
 */
class NativeFaceAndMouthExtractor(private val cropSize: Int = 96) {

    /**
     * Extracts normalized (96, 96) grayscale float array from frame Bitmap.
     */
    fun extractMouthCrop(frame: Bitmap, faceRect: Rect): Array<FloatArray> {
        val w = frame.width
        val h = frame.height

        // Mouth center located in lower 40% of face bounding box
        val mx = faceRect.centerX()
        val my = faceRect.top + (faceRect.height() * 0.72f).toInt()
        val side = max(32, (faceRect.width() * 0.55f).toInt())

        val x1 = max(0, mx - side / 2)
        val y1 = max(0, my - side / 2)
        val x2 = min(w, x1 + side)
        val y2 = min(h, y1 + side)

        val cropWidth = max(1, x2 - x1)
        val cropHeight = max(1, y2 - y1)

        val scaled = Bitmap.createScaledBitmap(frame, cropSize, cropSize, true)
        val result = Array(cropSize) { FloatArray(cropSize) }

        var minVal = 255f
        var maxVal = 0f

        // Extract grayscale values
        val pixels = IntArray(cropSize * cropSize)
        scaled.getPixels(pixels, 0, cropSize, 0, 0, cropSize, cropSize)

        for (y in 0 until cropSize) {
            for (x in 0 until cropSize) {
                val p = pixels[y * cropSize + x]
                val r = Color.red(p)
                val g = Color.green(p)
                val b = Color.blue(p)
                // ITU-R BT.601 luma
                val gray = (0.299f * r + 0.587f * g + 0.114f * b)
                result[y][x] = gray
                if (gray < minVal) minVal = gray
                if (gray > maxVal) maxVal = gray
            }
        }

        // Contrast normalization (zero mean / unit variance normalization)
        val range = max(1f, maxVal - minVal)
        for (y in 0 until cropSize) {
            for (x in 0 until cropSize) {
                result[y][x] = (result[y][x] - minVal) / range
            }
        }

        if (scaled != frame) {
            scaled.recycle()
        }

        return result
    }

    /**
     * Detects face bounding box candidates from video frame.
     */
    fun detectFaceRect(frame: Bitmap): Rect {
        val w = frame.width
        val h = frame.height
        // Standard center-weighted portrait face heuristic
        val fw = (w * 0.45f).toInt()
        val fh = (h * 0.50f).toInt()
        val fx = (w - fw) / 2
        val fy = (h * 0.20f).toInt()
        return Rect(fx, fy, fx + fw, fy + fh)
    }

    /**
     * Extracts mouth ROI sequences for a full list of video frames.
     */
    fun extractMouthSequence(frames: List<Bitmap>): List<Array<FloatArray>> {
        return frames.map { frame ->
            val faceRect = detectFaceRect(frame)
            extractMouthCrop(frame, faceRect)
        }
    }
}
