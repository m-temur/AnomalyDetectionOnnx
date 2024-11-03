package ai.onnxruntime.example.objectdetection

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import java.nio.FloatBuffer
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Typeface
import android.util.Log

class AnomalyDetectorHelper(
    private val context: Context,
    private val detectorListener: DetectorListener
) {
    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var isInitialized = false

    init {
        setupDetector()
    }

    private fun setupDetector() {
        try {
            ortEnvironment = OrtEnvironment.getEnvironment()
            val modelBytes = context.assets.open("model.onnx").use { it.readBytes() }
            val sessionOptions = OrtSession.SessionOptions()
            ortSession = ortEnvironment?.createSession(modelBytes, sessionOptions)
            isInitialized = true
        } catch (e: Exception) {
            detectorListener.onError("Detector failed to initialize: ${e.message}")
        }
    }

    fun detect(bitmap: Bitmap, imageRotation: Int) {
        if (!isInitialized) {
            setupDetector()
        }

        try {
            val inferenceTime = SystemClock.uptimeMillis()
            val result = processImageAndRunInference(bitmap, imageRotation)
            detectorListener.onResults(
                result,
                SystemClock.uptimeMillis() - inferenceTime,
                bitmap.height,
                bitmap.width
            )
        } catch (e: Exception) {
            detectorListener.onError("Detection failed: ${e.message}")
        }
    }

    private fun processImageAndRunInference(bitmap: Bitmap, rotation: Int): DetectionResult {
        // Handle rotation first
        val rotatedBitmap = if (rotation != 0) {
            val matrix = Matrix()
            matrix.postRotate(rotation.toFloat())
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } else {
            bitmap
        }

        val bitmapCopy = bitmap.copy(bitmap.config, true)

        // Create scaled bitmap
        val resizedBitmap = Bitmap.createScaledBitmap(
            bitmapCopy,
            MODEL_INPUT_WIDTH,
            MODEL_INPUT_HEIGHT,
            true
        )

        try {

            val floatValues = FloatArray(size = resizedBitmap.width * resizedBitmap.height * 3)
            val pixels = IntArray(size = resizedBitmap.width * resizedBitmap.height)
            resizedBitmap.getPixels(pixels, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)

            for (i in pixels.indices) {
                val pixel = pixels[i]
                // Convert the pixel to RGB channels
                floatValues[i * 3 + 0] = ((pixel shr 16 and 0xFF) / 255.0f)
                floatValues[i * 3 + 1] = ((pixel shr 8 and 0xFF) / 255.0f)
                floatValues[i * 3 + 2] = ((pixel and 0xFF) / 255.0f)
            }

            val chmBuffer = FloatArray(size = resizedBitmap.width * resizedBitmap.height * 3)
            for (y in 0 until resizedBitmap.height) {
                for (x in 0 until resizedBitmap.width) {
                    val index = y * resizedBitmap.width + x
                    chmBuffer[0 * resizedBitmap.width * resizedBitmap.height + index] = floatValues[index * 3 + 0] // R
                    chmBuffer[1 * resizedBitmap.width * resizedBitmap.height + index] = floatValues[index * 3 + 1] // G
                    chmBuffer[2 * resizedBitmap.width * resizedBitmap.height + index] = floatValues[index * 3 + 2] // B
                }
            }

            // Create string representation
            val stringBuilder = StringBuilder()
            stringBuilder.append("[")
            chmBuffer.forEachIndexed { index, value ->
                stringBuilder.append(String.format("%.2f", value))
                if (index < floatValues.size - 1) stringBuilder.append(", ")
            }
            val img_str = stringBuilder.append("]")

            Log.d("AnomalyDetector", "Image str" + img_str.toString())
            Log.d("AnomalyDetector", "Image str_wrapped" + FloatBuffer.wrap(chmBuffer))
            val tensorShape = longArrayOf(1, 3, resizedBitmap.height.toLong(), resizedBitmap.width.toLong()) // NCHW format

            val inputTensor = OnnxTensor.createTensor(
                OrtEnvironment.getEnvironment(),
                FloatBuffer.wrap(chmBuffer),
                tensorShape
            )


            // Run inference
            val output = ortSession?.run(mapOf("input" to inputTensor))
            // Process outputs
            val anomalyMapTensor = output?.get(0) as? OnnxTensor
            val scoreTensor = output?.get(1) as? OnnxTensor

            val anomalyMap = anomalyMapTensor?.floatBuffer?.array() ?: FloatArray(0)
            val rawScore = scoreTensor?.floatBuffer?.get(0) ?: 0f

            Log.d("Model Output:", "Score {$rawScore}")
            // Normalize anomaly map to [0, 1] range
            val minAnomaly = anomalyMap.minOrNull() ?: 0f
            val maxAnomaly = anomalyMap.maxOrNull() ?: 1f

            val normalizedMap = anomalyMap.map {
                ((it - minAnomaly) / (maxAnomaly - minAnomaly)).coerceIn(0f, 1f)
            }.toFloatArray()

            // Calculate threshold and percentage
            val normalizedThreshold = 0.5f  // Adjust this threshold as needed
            val anomalousPixelPercentage = normalizedMap.count {
                it > normalizedThreshold
            }.toFloat() / normalizedMap.size

            // Determine if anomalous based on both score and pixel percentage
            val isAnomaly = rawScore > IMAGE_THRESHOLD ||
                    anomalousPixelPercentage > ANOMALY_PIXEL_PERCENTAGE_THRESHOLD

            val predLabel = if (isAnomaly) "Anomalous" else "Normal"

            return DetectionResult(
                originalBitmap = rotatedBitmap,
                rawScore = rawScore,
                predLabel = predLabel,
                anomalyMap = normalizedMap,
                pixelThreshold = normalizedThreshold,
                anomalousPixelPercentage = anomalousPixelPercentage
            )

        } finally {
            resizedBitmap.recycle()
        }
    }

    companion object {
        private const val MODEL_INPUT_WIDTH = 224
        private const val MODEL_INPUT_HEIGHT = 224
        private const val IMAGE_THRESHOLD = 50  // Adjust based on your model
        private const val ANOMALY_PIXEL_PERCENTAGE_THRESHOLD = 0.3f  // Adjust as needed
        private const val MIN_SCORE = 0f
        private const val MAX_SCORE = 1f
    }
    private fun normalizeScore(value: Float): Float {
        return ((value - IMAGE_THRESHOLD) / (MAX_SCORE - MIN_SCORE) + 0.5f)
            .coerceIn(0f, 1f)
    }


    data class DetectionResult(
        val originalBitmap: Bitmap,
        val rawScore: Float,
        val predLabel: String,
        val anomalyMap: FloatArray,
        val pixelThreshold: Float,
        val anomalousPixelPercentage: Float
    ) {
        private var visualizedBitmap: Bitmap? = null


        fun visualize(): Bitmap {
            try {
                val width = originalBitmap.width
                val height = originalBitmap.height

                val outputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                val canvas = Canvas(outputBitmap)

                // Draw original image first
                canvas.drawBitmap(originalBitmap, 0f, 0f, null)

                // Draw heatmap overlay
                val mapSize = kotlin.math.sqrt(anomalyMap.size.toFloat()).toInt()
                val cellWidth = width.toFloat() / mapSize
                val cellHeight = height.toFloat() / mapSize

                val paint = Paint(Paint.ANTI_ALIAS_FLAG)
                paint.style = Paint.Style.FILL

                for (y in 0 until mapSize) {
                    for (x in 0 until mapSize) {
                        val index = y + (mapSize - 1 - x) * mapSize
                        val value = anomalyMap[index]

                        paint.color = getHeatMapColor(value)
                        paint.alpha = (value * 100).toInt().coerceIn(0, 100)

                        canvas.drawRect(
                            x * cellWidth,
                            y * cellHeight,
                            (x + 1) * cellWidth,
                            (y + 1) * cellHeight,
                            paint
                        )
                    }
                }

                // Draw text with improved visibility
                // Background for text
                val bgPaint = Paint().apply {
                    color = Color.BLACK
                    alpha = 160
                    style = Paint.Style.FILL
                }

                // Configure text paint
                val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color = Color.WHITE
                    textSize = height / 20f  // Smaller text size
                    typeface = Typeface.DEFAULT_BOLD
                    setShadowLayer(3f, 0f, 0f, Color.BLACK)
                    textAlign = Paint.Align.LEFT
                }

                val scoreText = String.format("%.2f", rawScore)
                val resultText = "$predLabel (Score: $scoreText)"

                // Calculate text position (top left)
                val padding = height / 50f
                val textX = padding
                val textY = padding + textPaint.textSize  // Position from top

                // Measure text for background
                val bounds = Rect()
                textPaint.getTextBounds(resultText, 0, resultText.length, bounds)

                // Draw background rectangle
                canvas.drawRect(
                    textX - padding,
                    textY - bounds.height() - padding,
                    textX + bounds.width() + padding,
                    textY + padding,
                    bgPaint
                )

                // Draw text
                canvas.drawText(resultText, textX, textY, textPaint)

                return outputBitmap

            } catch (e: Exception) {
                Log.e("AnomalyDetector", "Error in visualization: ${e.message}")
                return originalBitmap
            }
        }



        private fun getHeatMapColor(value: Float): Int {
            val normalizedValue = value.coerceIn(0f, 1f)
            return when {
                normalizedValue < 0.25f -> {
                    // Blue range
                    val blue = 255
                    val green = (normalizedValue * 4 * 255).toInt()
                    Color.rgb(0, green, blue)
                }
                normalizedValue < 0.5f -> {
                    // Cyan to green
                    val factor = (normalizedValue - 0.25f) * 4
                    val blue = ((1 - factor) * 255).toInt()
                    Color.rgb(0, 255, blue)
                }
                normalizedValue < 0.75f -> {
                    // Green to yellow
                    val factor = (normalizedValue - 0.5f) * 4
                    val red = (factor * 255).toInt()
                    Color.rgb(red, 255, 0)
                }
                else -> {
                    // Yellow to red
                    val factor = (normalizedValue - 0.75f) * 4
                    val green = ((1 - factor) * 255).toInt()
                    Color.rgb(255, green, 0)
                }
            }
        }

        fun recycle() {
            visualizedBitmap?.recycle()
            visualizedBitmap = null
        }
    }
    interface DetectorListener {
        fun onResults(
            result: DetectionResult,
            inferenceTime: Long,
            imageHeight: Int,
            imageWidth: Int
        )
        fun onError(error: String)
    }


}