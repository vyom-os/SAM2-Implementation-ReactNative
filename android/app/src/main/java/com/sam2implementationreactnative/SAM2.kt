package com.sam2implementationreactnative

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import androidx.core.graphics.get
import androidx.core.graphics.set
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.util.Collections
import java.util.EnumSet
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.LinkedList

class Decoder {

    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var ortSession: OrtSession
    private lateinit var maskOutputString: String
    private lateinit var scoresOutputString: String

    private lateinit var imageEmbeddingInputString: String
    private lateinit var highResFeature0InputString: String
    private lateinit var highResFeature1InputString: String
    private lateinit var pointCoordinatesInputString: String
    private lateinit var pointLabelsInputString: String
    private lateinit var maskInputString: String
    private lateinit var hasmaskInputString: String

    suspend fun init(modelPath: String, useFP16: Boolean = false, useXNNPack: Boolean = false) =
		withContext(Dispatchers.IO) {
			ortEnvironment = OrtEnvironment.getEnvironment()
			val options =
					OrtSession.SessionOptions().apply {
						if (useFP16) {
							addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
						}
						if (useXNNPack) {
							addXnnpack(mapOf("intra_op_num_threads" to "2"))
						}
					}
			ortSession = ortEnvironment.createSession(modelPath, options)
			val decoderInputStrings = ortSession.inputNames.toList()
			val decoderOutputStrings = ortSession.outputNames.toList()
			imageEmbeddingInputString = decoderInputStrings[0]
			highResFeature0InputString = decoderInputStrings[1]
			highResFeature1InputString = decoderInputStrings[2]
			pointCoordinatesInputString = decoderInputStrings[3]
			pointLabelsInputString = decoderInputStrings[4]
			maskInputString = decoderInputStrings[5]
			hasmaskInputString = decoderInputStrings[6]
			maskOutputString = decoderOutputStrings[0]
			scoresOutputString = decoderOutputStrings[1]
		}

    suspend fun execute(
            encoderResults: Encoder.EncoderResults,
            pointCoordinates: FloatBuffer,
            pointLabels: FloatBuffer,
            numLabels: Long,
            numPoints: Long,
            inputImage: Bitmap
    ): Bitmap =
            withContext(Dispatchers.Default) {
                val imgHeight = inputImage.height
                val imgWidth = inputImage.width

                val imageEmbeddingTensor =
					OnnxTensor.createTensor(
						ortEnvironment,
						encoderResults.imageEmbedding,
						longArrayOf(1, 256, 64, 64),
					)
                val highResFeature0Tensor =
					OnnxTensor.createTensor(
						ortEnvironment,
						encoderResults.highResFeature0,
						longArrayOf(1, 32, 256, 256),
					)
                val highResFeature1Tensor =
					OnnxTensor.createTensor(
						ortEnvironment,
						encoderResults.highResFeature1,
						longArrayOf(1, 64, 128, 128),
					)

                val pointCoordinatesTensor =
					OnnxTensor.createTensor(
						ortEnvironment,
						pointCoordinates,
						longArrayOf(numLabels, numPoints, 2),
					)
                val pointLabelsTensor =
                        OnnxTensor.createTensor(
                                ortEnvironment,
                                pointLabels,
                                longArrayOf(numLabels, numPoints),
                        )

                val maskTensor =
					OnnxTensor.createTensor(
						ortEnvironment,
						FloatBuffer.wrap(
								FloatArray(numLabels.toInt() * 1 * 256 * 256) { 0f }
						),
						longArrayOf(numLabels, 1, 256, 256),
					)
                val hasMaskTensor =
					OnnxTensor.createTensor(
						ortEnvironment,
						FloatBuffer.wrap(floatArrayOf(0.0f)),
						longArrayOf(1)
					)
                val origImageSizeTensor =
					OnnxTensor.createTensor(
						ortEnvironment,
						IntBuffer.wrap(intArrayOf(imgHeight, imgWidth)),
						longArrayOf(2)
					)
                val outputs =
					ortSession.run(
						mapOf(
							imageEmbeddingInputString to imageEmbeddingTensor,
							highResFeature0InputString to highResFeature0Tensor,
							highResFeature1InputString to highResFeature1Tensor,
							pointCoordinatesInputString to pointCoordinatesTensor,
							pointLabelsInputString to pointLabelsTensor,
							maskInputString to maskTensor,
							hasmaskInputString to hasMaskTensor,
							"orig_im_size" to origImageSizeTensor,
						)
					)
                val mask = (outputs[maskOutputString].get() as OnnxTensor).floatBuffer
				val scores = (outputs[scoresOutputString].get() as OnnxTensor).floatBuffer.array()
				val maskBitmap = Bitmap.createBitmap(imgWidth, imgHeight, Bitmap.Config.ARGB_8888)

				val combinedMask = FloatArray(imgHeight * imgWidth) { 0f }

				for (maskIdx in scores.indices) {
					val maskStartIndex = maskIdx * imgHeight * imgWidth
					
					for (i in 0 until imgHeight * imgWidth) {
						val maskValue = mask[maskStartIndex + i]
						if (maskValue > 0.1f) {
							combinedMask[i] = 1f 
						}
					}
				}

				for (i in 0 until imgHeight) {
					for (j in 0 until imgWidth) {
						val maskValue = combinedMask[j + i * imgWidth]
						maskBitmap.setPixel(j, i, if (maskValue > 0.5f) Color.argb(128, 255, 0, 0) else Color.TRANSPARENT)
					}
				}
                return@withContext maskBitmap
            }
		
    private fun blendColors(baseColor: Int, overlayColor: Int): Int {
        val alpha = Color.alpha(overlayColor)
        val red = (Color.red(baseColor) * (255 - alpha) + Color.red(overlayColor) * alpha) / 255
        val green =
                (Color.green(baseColor) * (255 - alpha) + Color.green(overlayColor) * alpha) / 255
        val blue = (Color.blue(baseColor) * (255 - alpha) + Color.blue(overlayColor) * alpha) / 255
        return Color.argb(255, red, green, blue)
    }
    suspend fun close() = withContext(Dispatchers.IO) {
        ortSession.close()
        ortEnvironment.close()
	}
}


class Encoder {

    data class EncoderResults(
        val imageEmbedding: FloatBuffer,
        val highResFeature0: FloatBuffer,
        val highResFeature1: FloatBuffer
    )

    private val inputDim = 1024
    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var ortSession: OrtSession
    private lateinit var inputName: String
    private lateinit var imageEmbeddingOutputString: String
    private lateinit var highResFeature0OutputString: String
    private lateinit var highResFeature1OutputString: String

    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std = floatArrayOf(0.229f, 0.224f, 0.225f)

    suspend fun init(
        modelPath: String, useFP16: Boolean = false, useXNNPack: Boolean = false
    ) = withContext(Dispatchers.IO) {
        ortEnvironment = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions().apply {
            if (useFP16) {
                addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
            }
            if (useXNNPack) {
                addXnnpack(
                    mapOf(
                        "intra_op_num_threads" to "2"
                    )
                )
            }
        }
        ortSession = ortEnvironment.createSession(modelPath, options)
        inputName = ortSession.inputNames.first()
        val outputStrings = ortSession.outputNames.toList()
        highResFeature0OutputString = outputStrings[0]
        highResFeature1OutputString = outputStrings[1]
        imageEmbeddingOutputString = outputStrings[2]
    }

    suspend fun execute(inputImage: Bitmap) = withContext(Dispatchers.IO) {
        val resizedImage = Bitmap.createScaledBitmap(
            inputImage, inputDim, inputDim, true
        )

        val imagePixels = FloatBuffer.allocate(1 * resizedImage.width * resizedImage.height * 3)
        imagePixels.rewind()
        for (i in 0 until resizedImage.height) {
            for (j in 0 until resizedImage.width) {
                imagePixels.put(
                    ((Color.red(resizedImage[j, i]).toFloat() / 255.0f) - mean[0]) / std[0]
                )
            }
        }
        for (i in 0 until resizedImage.height) {
            for (j in 0 until resizedImage.width) {
                imagePixels.put(
                    ((Color.blue(resizedImage[j, i]).toFloat() / 255.0f) - mean[1]) / std[1]
                )
            }
        }
        for (i in 0 until resizedImage.height) {
            for (j in 0 until resizedImage.width) {
                imagePixels.put(
                    ((Color.green(resizedImage[j, i]).toFloat() / 255.0f) - mean[2]) / std[2]
                )
            }
        }
        imagePixels.rewind()

        val imageTensor = OnnxTensor.createTensor(
            ortEnvironment,
            imagePixels,
            longArrayOf(1, 3, inputDim.toLong(), inputDim.toLong()),
        )
        val outputs = ortSession.run(mapOf(inputName to imageTensor))
        val highResFeature0 = outputs[highResFeature0OutputString].get() as OnnxTensor
        val highResFeature1 = outputs[highResFeature1OutputString].get() as OnnxTensor
        val imageEmbedding = outputs[imageEmbeddingOutputString].get() as OnnxTensor
        return@withContext EncoderResults(
            imageEmbedding.floatBuffer, highResFeature0.floatBuffer, highResFeature1.floatBuffer
        )
    }
    suspend fun close() = withContext(Dispatchers.IO) {
		ortSession.close()
		ortEnvironment.close()
	}
}
data class Point(val x: Int, val y: Int)
