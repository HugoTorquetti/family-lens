package com.familylens

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface
import com.facebook.react.bridge.*
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.sqrt

class FaceRecognitionModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    private val faceDetector: FaceDetector
    private var interpreter: Interpreter? = null

    companion object {
        private const val MODEL_FILE = "facenet_512.tflite"
        private const val INPUT_SIZE = 160      // FaceNet espera imagens 160x160
        private const val EMBEDDING_SIZE = 512  // Este modelo FaceNet gera vetores de 512 dimensões
    }

    init {
        // Ativa o modo preciso e os landmarks (pontos de referência: olhos, nariz, boca)
        // Os landmarks são necessários para alinhar o rosto antes de rodar o FaceNet
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setMinFaceSize(0.1f)
            .build()
        faceDetector = FaceDetection.getClient(options)

        try {
            val model = loadModelFile()
            interpreter = Interpreter(model)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    override fun getName(): String = "FaceRecognition"

    private fun loadModelFile(): MappedByteBuffer {
        val assetFileDescriptor = reactApplicationContext.assets.openFd(MODEL_FILE)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            assetFileDescriptor.startOffset,
            assetFileDescriptor.declaredLength
        )
    }

    /**
     * Carrega um Bitmap a partir de um URI e corrige a rotação usando os dados EXIF.
     *
     * Fotos tiradas em retrato no Android são salvas em landscape no arquivo,
     * com um dado EXIF dizendo "rotacione X graus". Se ignorarmos esse dado,
     * o ML Kit vê o rosto deitado e não consegue detectá-lo.
     */
    private fun loadBitmapFromUri(uriString: String): Bitmap? {
        return if (uriString.startsWith("content://")) {
            val uri = android.net.Uri.parse(uriString)
            val resolver = reactApplicationContext.contentResolver

            // Lê o bitmap
            val bitmap = resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
                ?: return null

            // Lê o EXIF em um segundo stream (o primeiro já foi fechado)
            val exif = resolver.openInputStream(uri)?.use { ExifInterface(it) }

            rotateBitmapIfNeeded(bitmap, exif)
        } else {
            val path = uriString.removePrefix("file://")
            val bitmap = BitmapFactory.decodeFile(path) ?: return null
            val exif = ExifInterface(path)
            rotateBitmapIfNeeded(bitmap, exif)
        }
    }

    /**
     * Aplica a rotação indicada pelo EXIF ao bitmap.
     * A maioria das fotos de celular precisam de 90° de correção.
     */
    private fun rotateBitmapIfNeeded(bitmap: Bitmap, exif: ExifInterface?): Bitmap {
        val orientation = exif?.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL
        ) ?: ExifInterface.ORIENTATION_NORMAL

        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90  -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.preScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL   -> matrix.preScale(1f, -1f)
            else -> return bitmap  // Sem rotação necessária
        }

        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    /**
     * Alinha o rosto usando as posições dos olhos detectadas pelo ML Kit.
     *
     * O FaceNet foi treinado com rostos alinhados: olhos sempre na horizontal,
     * na mesma posição relativa dentro do recorte. Sem esse alinhamento, a orientação
     * da foto afeta o embedding e as comparações ficam imprecisas.
     *
     * O que fazemos:
     * 1. Calculamos o ângulo entre o olho esquerdo e o direito
     * 2. Rotacionamos a imagem inteira para que os olhos fiquem na horizontal
     * 3. Recortamos o rosto do bitmap já rotacionado
     */
    private fun alignFace(
        bitmap: Bitmap,
        leftEyeX: Float, leftEyeY: Float,
        rightEyeX: Float, rightEyeY: Float,
        bounds: android.graphics.Rect
    ): Bitmap {
        // atan2 retorna o ângulo em radianos entre o vetor (olho direito - olho esquerdo) e o eixo X
        // Convertemos para graus para usar na Matrix do Android
        val angle = Math.toDegrees(
            atan2((rightEyeY - leftEyeY).toDouble(), (rightEyeX - leftEyeX).toDouble())
        ).toFloat()

        // Ponto de pivô da rotação: centro do rosto
        val pivotX = (bounds.left + bounds.right) / 2f
        val pivotY = (bounds.top + bounds.bottom) / 2f

        // Matrix é a classe do Android para transformações geométricas em bitmaps
        val matrix = Matrix()
        matrix.postRotate(-angle, pivotX, pivotY)

        // Cria um novo bitmap com a imagem inteira rotacionada
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

        // Recorta o rosto do bitmap rotacionado, garantindo que não saia das bordas
        val left   = maxOf(0, bounds.left)
        val top    = maxOf(0, bounds.top)
        val right  = minOf(rotated.width, bounds.right)
        val bottom = minOf(rotated.height, bounds.bottom)

        val cropped = Bitmap.createBitmap(rotated, left, top, right - left, bottom - top)
        return Bitmap.createScaledBitmap(cropped, INPUT_SIZE, INPUT_SIZE, true)
    }

    @ReactMethod
    fun generateEmbedding(imageUri: String, promise: Promise) {
        try {
            val bitmap = loadBitmapFromUri(imageUri)
                ?: return promise.reject("ERROR", "Não foi possível carregar a imagem: $imageUri")

            val image = InputImage.fromBitmap(bitmap, 0)

            faceDetector.process(image)
                .addOnSuccessListener { faces ->
                    if (faces.isEmpty()) {
                        promise.reject("NO_FACE", "Nenhum rosto detectado na imagem")
                        return@addOnSuccessListener
                    }

                    // Pega o rosto com maior área — ignora rostos pequenos de fundo,
                    // brinquedos, mascotes ou pessoas em segundo plano
                    val face = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }!!
                    val bounds = face.boundingBox

                    // Tenta obter os landmarks dos olhos para alinhar o rosto
                    val leftEye  = face.getLandmark(FaceLandmark.LEFT_EYE)
                    val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)

                    val faceBitmap: Bitmap = if (leftEye != null && rightEye != null) {
                        // Temos os dois olhos: alinha o rosto antes de recortar
                        alignFace(
                            bitmap,
                            leftEye.position.x,  leftEye.position.y,
                            rightEye.position.x, rightEye.position.y,
                            bounds
                        )
                    } else {
                        // Fallback: sem landmarks, faz só o recorte simples
                        val left   = maxOf(0, bounds.left)
                        val top    = maxOf(0, bounds.top)
                        val right  = minOf(bitmap.width, bounds.right)
                        val bottom = minOf(bitmap.height, bounds.bottom)

                        if (right <= left || bottom <= top) {
                            promise.reject("ERROR", "Bounding box do rosto inválido")
                            return@addOnSuccessListener
                        }

                        val cropped = Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top)
                        Bitmap.createScaledBitmap(cropped, INPUT_SIZE, INPUT_SIZE, true)
                    }

                    val embedding = runFaceNet(faceBitmap)

                    val result = Arguments.createArray()
                    embedding.forEach { result.pushDouble(it.toDouble()) }
                    promise.resolve(result)
                }
                .addOnFailureListener { e ->
                    promise.reject("DETECTION_ERROR", e.message)
                }
        } catch (e: Exception) {
            promise.reject("ERROR", e.message)
        }
    }

    /**
     * Prepara a imagem e roda o modelo TFLite.
     *
     * Usa normalização por imagem (whitening): subtrai a média e divide pelo desvio padrão
     * calculados a partir dos próprios pixels da imagem.
     */
    private fun runFaceNet(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val floatPixels = FloatArray(INPUT_SIZE * INPUT_SIZE * 3)
        var idx = 0
        for (pixel in pixels) {
            floatPixels[idx++] = (pixel shr 16 and 0xFF).toFloat()
            floatPixels[idx++] = (pixel shr 8  and 0xFF).toFloat()
            floatPixels[idx++] = (pixel        and 0xFF).toFloat()
        }

        val mean = floatPixels.average().toFloat()
        val variance = floatPixels.map { (it - mean) * (it - mean) }.average().toFloat()
        val std = sqrt(variance)
        val adjustedStd = max(std, 1f / sqrt(floatPixels.size.toFloat()))

        val byteBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        byteBuffer.order(ByteOrder.nativeOrder())
        for (value in floatPixels) {
            byteBuffer.putFloat((value - mean) / adjustedStd)
        }

        val output = Array(1) { FloatArray(EMBEDDING_SIZE) }
        interpreter?.run(byteBuffer, output)

        return output[0]
    }

    /**
     * Compara dois embeddings e retorna a distância cosseno entre eles.
     *
     * Distância cosseno = 1 - similaridade cosseno
     * Valores próximos de 0 → mesma pessoa
     * Valores acima de ~0.6 → pessoas diferentes
     */
    @ReactMethod
    fun compareEmbeddings(emb1: ReadableArray, emb2: ReadableArray, promise: Promise) {
        try {
            val v1 = FloatArray(emb1.size()) { emb1.getDouble(it).toFloat() }
            val v2 = FloatArray(emb2.size()) { emb2.getDouble(it).toFloat() }

            var dot   = 0f
            var norm1 = 0f
            var norm2 = 0f

            for (i in v1.indices) {
                dot   += v1[i] * v2[i]
                norm1 += v1[i] * v1[i]
                norm2 += v2[i] * v2[i]
            }

            val cosineSimilarity = dot / (sqrt(norm1) * sqrt(norm2))
            val cosineDistance   = 1f - cosineSimilarity

            promise.resolve(cosineDistance.toDouble())
        } catch (e: Exception) {
            promise.reject("ERROR", e.message)
        }
    }
}
