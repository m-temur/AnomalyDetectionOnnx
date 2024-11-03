package ai.onnxruntime.example.objectdetection

import ai.onnxruntime.example.objectdetection.fragments.CameraFragment
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import android.Manifest
import android.app.ActivityManager
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Build
import android.widget.Toast
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import android.util.Log
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.widget.ImageView
import android.widget.TextView
import java.io.ByteArrayOutputStream

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (savedInstanceState == null) {
            supportFragmentManager.beginTransaction()
                .replace(R.id.fragment_container, CameraFragment())
                .commitAllowingStateLoss()
        }
    }

    override fun onDestroy() {
        // Make sure to clean up resources
        try {
            supportFragmentManager.fragments.forEach { fragment ->
                supportFragmentManager.beginTransaction()
                    .remove(fragment)
                    .commitAllowingStateLoss()
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Error cleaning up fragments: ${e.message}")
        }
        super.onDestroy()
    }
}