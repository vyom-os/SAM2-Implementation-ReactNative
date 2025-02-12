package com.sam2implementationreactnative

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.util.Log
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableArray
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.FloatBuffer
import kotlinx.coroutines.*

class SAMModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    private val encoder = Encoder()
    private val decoder = Decoder()
    private val encoderFileName = "tiny_encoder.onnx"
    private val decoderFileName = "tiny_decoder.onnx"
    private var lastEncoderResults: Encoder.EncoderResults? = null
    private var lastBitmap: Bitmap? = null
    private val masksByLabel = mutableMapOf<Int, Bitmap>()
    private val pointsByLabel = mutableMapOf<Int, MutableList<Point>>()
    private val labelsByLabelId = mutableMapOf<Int, MutableList<Float>>()

    data class Point(val x: Float, val y: Float)

    override fun getName(): String {
        return "SAMModule"
    }

    @ReactMethod
    fun initializeModels(promise: Promise) {
        CoroutineScope(Dispatchers.Main).launch {
            try {
                if (isModelInAssets(encoderFileName) && isModelInAssets(decoderFileName)) {
                    copyModelToStorage(encoderFileName)
                    copyModelToStorage(decoderFileName)
                    encoder.init(File(reactContext.filesDir, encoderFileName).absolutePath)
                    decoder.init(File(reactContext.filesDir, decoderFileName).absolutePath)
                }
                promise.resolve(true)
            } catch (e: Exception) {
                promise.reject("INIT_ERROR", e.message)
            }
        }
    }

    private fun isModelInAssets(modelFileName: String): Boolean {
        return (reactContext.assets.list("") ?: emptyArray()).contains(modelFileName)
    }

    private fun copyModelToStorage(modelFileName: String) {
        val modelFile = File(reactContext.filesDir, modelFileName)
        if (!modelFile.exists()) {
            reactContext.assets.open(modelFileName).use { inputStream ->
                reactContext.openFileOutput(modelFileName, Context.MODE_PRIVATE).use { outputStream
                    ->
                    inputStream.copyTo(outputStream)
                }
            }
        }
    }

    @ReactMethod
    fun processImage(imageBase64: String, points: ReadableArray, labelId: Int, promise: Promise) {
        CoroutineScope(Dispatchers.Main).launch {
            try {
                val cleanBase64 =
                    imageBase64
                        .replace("data:image/jpeg;base64,", "")
                        .replace("data:image/png;base64,", "")
                        .replace("\n", "")
                        .replace(" ", "")
                val imageBytes =
                    android.util.Base64.decode(cleanBase64, android.util.Base64.DEFAULT)
                val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                lastBitmap = bitmap

                val encoderResults =
                    if (lastEncoderResults == null) {
                        encoder.execute(bitmap).also { lastEncoderResults = it }
                    } else {
                        lastEncoderResults!!
                    }

                pointsByLabel[labelId] = mutableListOf()
                labelsByLabelId[labelId] = mutableListOf()

                val pointsSize = points.size()
                for (i in 0 until pointsSize) {
                    val point = points.getMap(i)
                    val modelX = (point.getDouble("x").toFloat() * 1024)
                    val modelY = (point.getDouble("y").toFloat() * 1024)

                    pointsByLabel[labelId]?.add(Point(modelX, modelY))
                    labelsByLabelId[labelId]?.add(point.getInt("label").toFloat())
                }

                val currentPoints = pointsByLabel[labelId] ?: mutableListOf()
                val currentLabels = labelsByLabelId[labelId] ?: mutableListOf()
                val pointsBuffer = FloatBuffer.allocate(currentPoints.size * 2)
                val labelsBuffer = FloatBuffer.allocate(currentPoints.size)

                currentPoints.forEach { point ->
                    pointsBuffer.put(point.x)
                    pointsBuffer.put(point.y)
                }
                currentLabels.forEach { label -> labelsBuffer.put(label) }

                pointsBuffer.rewind()
                labelsBuffer.rewind()

                val maskBitmap =
                    decoder.execute(
                        encoderResults,
                        pointsBuffer,
                        labelsBuffer,
                        1,
                        currentPoints.size.toLong(),
                        bitmap,
                    )

                masksByLabel[labelId] = maskBitmap

                val combinedMask = combineMasks(bitmap)

                val byteArrayOutputStream = ByteArrayOutputStream()
                combinedMask.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream)
                val maskBytes = byteArrayOutputStream.toByteArray()
                val maskBase64 =
                    android.util.Base64.encodeToString(maskBytes, android.util.Base64.DEFAULT)
                promise.resolve(maskBase64)
            } catch (e: Exception) {
                Log.d("SAM2", "Error processing image: ${e.message}")
                promise.reject("PROCESS_ERROR", e.message)
            }
        }
    }

    private fun combineMasks(originalBitmap: Bitmap): Bitmap {
        val combined = originalBitmap.copy(Bitmap.Config.ARGB_8888, true)

        masksByLabel.forEach { (labelId, maskBitmap) ->
            val hue = (labelId * 137.5f) % 360f
            val color = android.graphics.Color.HSVToColor(128, floatArrayOf(hue, 1f, 1f))

            for (x in 0 until combined.width) {
                for (y in 0 until combined.height) {
                    if (Color.alpha(maskBitmap.getPixel(x, y)) > 0) {
                        combined.setPixel(x, y, blendColors(combined.getPixel(x, y), color))
                    }
                }
            }
        }
        return combined
    }

    @ReactMethod
    fun clearPoints(promise: Promise) {
        CoroutineScope(Dispatchers.Main).launch {
            // encoder.close()
            // decoder.close()
            lastEncoderResults = null
            masksByLabel.clear()
            pointsByLabel.clear()
            labelsByLabelId.clear()
            lastBitmap = null
            promise.resolve(true)
        }
    }

    private fun blendColors(baseColor: Int, overlayColor: Int): Int {
        val alpha = Color.alpha(overlayColor)
        val red = (Color.red(baseColor) * (255 - alpha) + Color.red(overlayColor) * alpha) / 255
        val green =
            (Color.green(baseColor) * (255 - alpha) + Color.green(overlayColor) * alpha) / 255
        val blue = (Color.blue(baseColor) * (255 - alpha) + Color.blue(overlayColor) * alpha) / 255
        return Color.argb(255, red, green, blue)
    }

    @ReactMethod
    fun removeLabel(labelId: Int, promise: Promise) {
        masksByLabel.remove(labelId)
        pointsByLabel.remove(labelId)
        labelsByLabelId.remove(labelId)

        lastBitmap?.let { bitmap ->
            val combinedMask = combineMasks(bitmap)
            val byteArrayOutputStream = ByteArrayOutputStream()
            combinedMask.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream)
            val maskBytes = byteArrayOutputStream.toByteArray()
            val maskBase64 =
                android.util.Base64.encodeToString(maskBytes, android.util.Base64.DEFAULT)
            promise.resolve(maskBase64)
        } ?: promise.resolve(null)
    }
}

