package ai.onnxruntime.example.objectdetection

import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import org.json.JSONObject
import java.io.InputStream
import java.nio.FloatBuffer
import android.content.pm.PackageManager
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
fun JSONObject.toMap(): Map<String, Any> = keys().asSequence().associateWith { get(it) }

class AnomalyDetector(private val context: Context, private val ortSession: OrtSession) {
    private lateinit var metadata: Map<String, Any>
    private lateinit var inputShape: LongArray
    private lateinit var inputName: String
    private var imageThreshold: Float = 0f
    private var pixelThreshold: Float = 0f
    private var minScore: Float = 0f
    private var maxScore: Float = 0f

    init {
        loadMetadata()
        initializeInputInfo()
    }

    private fun loadMetadata() {
        val jsonString = context.assets.open("metadata.json").bufferedReader().use { it.readText() }
        metadata = JSONObject(jsonString).toMap()
        imageThreshold = (metadata["image_threshold"] as? Double)?.toFloat() ?: 42.5799674987793f
        pixelThreshold = (metadata["pixel_threshold"] as? Double)?.toFloat() ?: 42.5799674987793f
        minScore = (metadata["pred_scores_min"] as? Double)?.toFloat() ?: 54.49655514941406f
        maxScore = (metadata["pred_scores_max"] as? Double)?.toFloat() ?: 70.5367202758789f
    }

    private fun initializeInputInfo() {
        val inputInfo = ortSession.inputInfo
        if (inputInfo.isEmpty()) {
            throw IllegalStateException("Model has no input")
        }
        val firstInput = inputInfo.entries.first()
        inputName = firstInput.key
        val tensorInfo = firstInput.value.info as? TensorInfo
            ?: throw IllegalStateException("Input is not a tensor")
        inputShape = tensorInfo.shape
    }

    fun detect(inputStream: InputStream): Result {
        val bitmap = BitmapFactory.decodeStream(inputStream)
        //val inputTensorData = preprocessImage(bitmap, inputShape[2].toInt(), inputShape[3].toInt())

        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputShape[2].toInt(), inputShape[3].toInt(), true)
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

        try {
            val output = ortSession.run(mapOf("input" to inputTensor))
            return postProcess(output, bitmap)
        } finally {
            inputTensor.close()
        }
    }

    fun detectFromBitmap(bitmap: Bitmap): Result {
        var resizedBitmap: Bitmap? = null
        var inputTensor: OnnxTensor? = null

        try {
            // Create a copy of the input bitmap that we can resize
            resizedBitmap = Bitmap.createScaledBitmap(
                bitmap,
                inputShape[2].toInt(),
                inputShape[3].toInt(),
                true
            )

            val floatValues = FloatArray(size = resizedBitmap.width * resizedBitmap.height * 3)
            val pixels = IntArray(size = resizedBitmap.width * resizedBitmap.height)
            resizedBitmap.getPixels(pixels, 0, resizedBitmap.width, 0, 0,
                resizedBitmap.width, resizedBitmap.height)

            // Convert to CHW format (don't modify original pixels array)
            val chmBuffer = FloatArray(size = resizedBitmap.width * resizedBitmap.height * 3)
            for (y in 0 until resizedBitmap.height) {
                for (x in 0 until resizedBitmap.width) {
                    val pixelIndex = y * resizedBitmap.width + x
                    val pixel = pixels[pixelIndex]

                    // Store in CHW format
                    val h = resizedBitmap.height
                    val w = resizedBitmap.width
                    chmBuffer[0 * h * w + pixelIndex] = ((pixel shr 16 and 0xFF) / 255.0f)  // R
                    chmBuffer[1 * h * w + pixelIndex] = ((pixel shr 8 and 0xFF) / 255.0f)   // G
                    chmBuffer[2 * h * w + pixelIndex] = ((pixel and 0xFF) / 255.0f)         // B
                }
            }

            val tensorShape = longArrayOf(1, 3, resizedBitmap.height.toLong(), resizedBitmap.width.toLong())
            inputTensor = OnnxTensor.createTensor(
                OrtEnvironment.getEnvironment(),
                FloatBuffer.wrap(chmBuffer),
                tensorShape
            )

            val output = ortSession.run(mapOf("input" to inputTensor))

            // Create a copy of the original bitmap for the result
            val resultBitmap = bitmap.copy(bitmap.config, true)
            return postProcess(output, resultBitmap)

        } finally {
            // Clean up resources
            inputTensor?.close()
            resizedBitmap?.recycle()
        }
    }


    /*
    private fun preprocessImage(bitmap: Bitmap, targetHeight: Int, targetWidth: Int): Bitmap {
        val resizedBitmap = bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
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

        return resizedBitmap
    }
*/
    private fun postProcess(output: OrtSession.Result, inputBitmap: Bitmap): Result {
        val anomalyMapTensor = output.get(0) as OnnxTensor
        val scoreTensor = output.get(1) as OnnxTensor

        val anomalyMap = anomalyMapTensor.floatBuffer.array()
        var rawScore = scoreTensor.floatBuffer.get(0)
        val minAnomaly = anomalyMap.minOrNull() ?: 0f
        val maxAnomaly = anomalyMap.maxOrNull() ?: 1f
        // Normalize anomaly scores
        val normalizedMap = anomalyMap.map { (it - minAnomaly) / (maxAnomaly - minAnomaly) }
        // Calculate the percentage of pixels above the normalized threshold
        val normalizedThreshold = (pixelThreshold - minAnomaly) / (maxAnomaly - minAnomaly)
        val anomalousPixelPercentage = normalizedMap.count { it > normalizedThreshold }.toFloat() / normalizedMap.size

        // Classify as anomalous if the raw score is above the image threshold
        val isAnomaly = rawScore > imageThreshold
        val predLabel = if (isAnomaly) "Anomalous" else "Normal"
        //rawScore = if (isAnomaly) rawScore else (100.0f-rawScore)
        // Apply normalization
        val normalizedScore = normalize(rawScore, imageThreshold, minScore, maxScore)

        Log.d("AnomalyDetector", "Raw score: $normalizedScore")
        Log.d("AnomalyDetector", "Anomaly map size: ${anomalyMap.size}")
        Log.d("AnomalyDetector", "Anomaly map min: $minAnomaly, max: $maxAnomaly")
        Log.d("AnomalyDetector", "Image threshold: $imageThreshold")
        Log.d("AnomalyDetector", "Pixel threshold: $pixelThreshold")
        Log.d("AnomalyDetector", "Anomalous pixel percentage: $anomalousPixelPercentage")
        Log.d("AnomalyDetector", "Is Anomaly: $isAnomaly")
        Log.d("AnomalyDetector", "Prediction Label: $predLabel")

        return Result(
            originalBitmap = inputBitmap,
            rawScore = normalizedScore,
            predLabel = predLabel,
            anomalyMap = normalizedMap.toFloatArray(),
            pixelThreshold = normalizedThreshold,
            anomalousPixelPercentage = anomalousPixelPercentage
        )
    }

    fun normalize(
        value: Float,
        threshold: Float,
        minVal: Float,
        maxVal: Float
    ): Float {
        // Apply min-max normalization and shift the values such that the threshold value is centered at 0.5
        val normalized = ((value - threshold) / (maxVal - minVal)) + 0.5f

        // Clip values to range [0, 1]
        return normalized.coerceIn(0f, 1f)
    }


    data class Result(
        val originalBitmap: Bitmap, // Changed from private to public
        val rawScore: Float,
        val predLabel: String,
        val anomalyMap: FloatArray,
        val pixelThreshold: Float,
        val anomalousPixelPercentage: Float
    ) {
        private var visualizedBitmap: Bitmap? = null

        fun visualize(): Bitmap {
            visualizedBitmap?.let {
                if (!it.isRecycled) return it
            }

            val width = originalBitmap.width
            val height = originalBitmap.height

            val outputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(outputBitmap)

            // Draw original image
            canvas.drawBitmap(originalBitmap, 0f, 0f, null)

            // Draw heatmap
            val paint = Paint()
            val mapSize = kotlin.math.sqrt(anomalyMap.size.toFloat()).toInt()
            val cellWidth = width.toFloat() / mapSize
            val cellHeight = height.toFloat() / mapSize

            for (y in 0 until mapSize) {
                for (x in 0 until mapSize) {
                    val index = y * mapSize + x
                    val value = anomalyMap[index]

                    paint.color = getHeatMapColor(value)
                    paint.alpha = (value * 128).toInt().coerceIn(0, 128)  // 50% max opacity

                    val left = x * cellWidth
                    val top = y * cellHeight
                    val right = left + cellWidth
                    val bottom = top + cellHeight

                    canvas.drawRect(left, top, right, bottom, paint)
                }
            }

            // Draw text overlay
            val textPaint = Paint().apply {
                color = Color.YELLOW
                textSize = (height / 20f).coerceAtLeast(24f)
                style = Paint.Style.FILL
                setShadowLayer(2f, 0f, 0f, Color.BLACK)
                textAlign = Paint.Align.LEFT
            }

            // Format score as percentage
            val scoreText = if (predLabel == "Anomalous") {
                String.format("%.1f%%", rawScore * 100)
            } else {
                String.format("%.1f%%", (1 - rawScore) * 100)
            }

            val resultText = "$predLabel ($scoreText)"
            val textX = 20f
            val textY = height - 30f
            canvas.drawText(resultText, textX, textY, textPaint)

            visualizedBitmap = outputBitmap
            return outputBitmap
        }

        private fun getHeatMapColor(value: Float): Int {
            val normalizedValue = value.coerceIn(0f, 1f)
            return when {
                normalizedValue < 0.25f -> Color.rgb(0, 0, (normalizedValue * 4 * 255).toInt())
                normalizedValue < 0.5f -> Color.rgb(0, ((normalizedValue - 0.25f) * 4 * 255).toInt(), 255)
                normalizedValue < 0.75f -> Color.rgb(((normalizedValue - 0.5f) * 4 * 255).toInt(), 255,
                    ((0.75f - normalizedValue) * 4 * 255).toInt())
                else -> Color.rgb(255, ((1f - normalizedValue) * 4 * 255).toInt(), 0)
            }
        }
    }

}
