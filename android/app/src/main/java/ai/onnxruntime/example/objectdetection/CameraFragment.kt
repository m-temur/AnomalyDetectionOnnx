package ai.onnxruntime.example.objectdetection.fragments

import ai.onnxruntime.example.objectdetection.AnomalyDetectorHelper
import ai.onnxruntime.example.objectdetection.databinding.FragmentCameraBinding
import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraFragment : Fragment(), AnomalyDetectorHelper.DetectorListener {
    private var _binding: FragmentCameraBinding? = null
    private val binding get() = _binding!!
    private var lastBitmap: Bitmap? = null

    private var anomalyDetectorHelper: AnomalyDetectorHelper? = null
    private var cameraExecutor: ExecutorService? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var isProcessingImage = false // Add this flag
    private var currentResult: AnomalyDetectorHelper.DetectionResult? = null


    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentCameraBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Initialize camera executor
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Initialize detector
        initializeDetector()

        // Start camera if permissions are granted
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions(REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
    }

    private fun initializeDetector() {
        try {
            anomalyDetectorHelper = AnomalyDetectorHelper(
                requireContext(),
                this
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing detector: ${e.message}")
            showToast("Failed to initialize detector")
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())

        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                bindCameraUseCases()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get camera provider: ${e.message}")
                showToast("Failed to start camera")
            }
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    private fun bindCameraUseCases() {
        try {
            val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

            // Preview use case
            val preview = Preview.Builder()
                .setTargetRotation(binding.viewFinder.display.rotation)
                .build()

            // Image analysis use case
            imageAnalyzer = ImageAnalysis.Builder()
                .setTargetRotation(binding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()
                .also { analysis ->
                    cameraExecutor?.let {
                        analysis.setAnalyzer(it) { imageProxy ->
                            try {
                                processImage(imageProxy)
                            } catch (e: Exception) {
                                Log.e(TAG, "Error processing image: ${e.message}")
                            } finally {
                                imageProxy.close()
                            }
                        }
                    }
                }

            try {
                // Must unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                camera = cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalyzer
                )

                // Attach the preview to PreviewView
                preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)

            } catch (e: Exception) {
                Log.e(TAG, "Use case binding failed: ${e.message}")
                // Show error to user
                Toast.makeText(
                    requireContext(),
                    "Camera initialization failed. Please restart the app.",
                    Toast.LENGTH_LONG
                ).show()
            }

        } catch (e: Exception) {
            Log.e(TAG, "Camera setup failed: ${e.message}")
        }
    }

    private fun showToast(message: String) {
        activity?.runOnUiThread {
            Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
        }
    }

    override fun onPause() {
        super.onPause()
        try {
            cameraProvider?.unbindAll()
        } catch (e: Exception) {
            Log.e(TAG, "Error unbinding camera uses cases: ${e.message}")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            cameraExecutor?.shutdown()
            cameraExecutor = null
        } catch (e: Exception) {
            Log.e(TAG, "Error shutting down camera executor: ${e.message}")
        }
    }

    private fun processImage(imageProxy: ImageProxy) {
        if (isProcessingImage) {
            imageProxy.close()
            return
        }

        try {
            isProcessingImage = true

            // Clean up previous result
            currentResult?.recycle()
            currentResult = null

            val bitmap = imageProxyToBitmap(imageProxy)
            bitmap?.let {
                try {
                    anomalyDetectorHelper?.detect(it, imageProxy.imageInfo.rotationDegrees)
                } finally {
                    bitmap.recycle() // Recycle the temporary bitmap
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing image: ${e.message}")
        } finally {
            isProcessingImage = false
            imageProxy.close()
        }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        try {
            val yBuffer = imageProxy.planes[0].buffer
            val uBuffer = imageProxy.planes[1].buffer
            val vBuffer = imageProxy.planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)

            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = YuvImage(nv21, ImageFormat.NV21,
                imageProxy.width, imageProxy.height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(
                Rect(0, 0, imageProxy.width, imageProxy.height),
                75, // Reduced quality for better performance
                out
            )
            val imageBytes = out.toByteArray()

            // Calculate target size while maintaining aspect ratio
            val (targetWidth, targetHeight) = calculateTargetSize(
                imageProxy.width,
                imageProxy.height,
                TARGET_SIZE
            )

            // Decode with scaled dimensions
            val options = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size, options)

            options.apply {
                inJustDecodeBounds = false
                inSampleSize = calculateInSampleSize(options, targetWidth, targetHeight)
            }

            return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size, options)
        } catch (e: Exception) {
            Log.e(TAG, "Error converting image: ${e.message}")
            return null
        }
    }

    private fun calculateTargetSize(width: Int, height: Int, targetSize: Int): Pair<Int, Int> {
        val ratio = width.toFloat() / height.toFloat()
        return if (width > height) {
            Pair(targetSize, (targetSize / ratio).toInt())
        } else {
            Pair((targetSize * ratio).toInt(), targetSize)
        }
    }

    private fun calculateInSampleSize(
        options: BitmapFactory.Options,
        reqWidth: Int,
        reqHeight: Int
    ): Int {
        val (height: Int, width: Int) = options.run { outHeight to outWidth }
        var inSampleSize = 1

        if (height > reqHeight || width > reqWidth) {
            val halfHeight: Int = height / 2
            val halfWidth: Int = width / 2

            while (halfHeight / inSampleSize >= reqHeight &&
                halfWidth / inSampleSize >= reqWidth) {
                inSampleSize *= 2
            }
        }
        return inSampleSize
    }


    override fun onResults(
        result: AnomalyDetectorHelper.DetectionResult,
        inferenceTime: Long,
        imageHeight: Int,
        imageWidth: Int
    ) {
        if (!isAdded) return

        currentResult = result

        activity?.runOnUiThread {
            try {
                binding.outputImageView.setImageBitmap(result.visualize())
            } catch (e: Exception) {
                Log.e(TAG, "Error updating UI: ${e.message}")
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        currentResult?.recycle()
        currentResult = null
        _binding = null
    }

    override fun onError(error: String) {
        if (!isAdded) return

        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
        }
    }

    companion object {
        private const val TAG = "CameraFragment"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private const val TARGET_SIZE = 224 // Your model's input size
    }



    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(requireContext(), it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(
                    requireContext(),
                    "Permissions not granted.",
                    Toast.LENGTH_SHORT
                ).show()
                requireActivity().finish()
            }
        }
    }

}