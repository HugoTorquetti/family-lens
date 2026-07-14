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
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.atan2
import kotlin.math.sqrt

class FaceRecognitionModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    private val faceDetector: FaceDetector

    // ONNX Runtime: ambiente global (reutilizado entre chamadas) e sessão do modelo
    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null

    companion object {
        private const val MODEL_FILE = "w600k_r50.onnx"  // InsightFace ResNet50 — WebFace600K
        private const val INPUT_SIZE = 112                // Entrada 112×112 (mesmo padrão ArcFace)
        private const val EMBEDDING_SIZE = 512            // ResNet50 gera vetores de 512 dimensões
    }

    init {
        // Ativa o modo preciso e os landmarks (pontos de referência: olhos, nariz, boca)
        // Os landmarks são necessários para alinhar o rosto antes de rodar o FaceNet
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            // CLASSIFICATION_MODE_ALL habilita leftEyeOpenProbability e rightEyeOpenProbability
            // Usamos isso para filtrar rostos de bonecas/mascotes, que não têm olhos "reais"
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .setMinFaceSize(0.05f)
            .build()
        faceDetector = FaceDetection.getClient(options)

        try {
            // Carrega o modelo via caminho de arquivo (não via bytes) para evitar OOM.
            // readBytes() tentaria alocar 174MB no heap Java de uma só vez, causando crash.
            // Ao usar o caminho, o ONNX Runtime faz memory-mapping nativo — muito mais eficiente.
            val modelPath = getOrCopyModelToFiles()
            ortSession = ortEnv.createSession(modelPath, OrtSession.SessionOptions())
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    override fun getName(): String = "FaceRecognition"

    /**
     * Copia o modelo de assets/ para o armazenamento interno do app (filesDir)
     * e retorna o caminho absoluto do arquivo.
     *
     * Na primeira execução: copia o arquivo (pode levar alguns segundos para 174MB).
     * Nas execuções seguintes: retorna o caminho direto sem copiar de novo.
     *
     * Por que não carregar diretamente de assets?
     * Assets não têm um caminho de arquivo real no sistema de arquivos — são lidos
     * via AssetManager. O ONNX Runtime precisa de um caminho real para fazer mmap().
     */
    private fun getOrCopyModelToFiles(): String {
        val dest = java.io.File(reactApplicationContext.filesDir, MODEL_FILE)
        if (!dest.exists()) {
            reactApplicationContext.assets.open(MODEL_FILE).use { input ->
                java.io.FileOutputStream(dest).use { output ->
                    input.copyTo(output, bufferSize = 8 * 1024 * 1024) // buffer de 8MB
                }
            }
        }
        return dest.absolutePath
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
     * O InsightFace foi treinado com rostos alinhados: olhos sempre na horizontal.
     *
     * Sequência correta:
     * 1. Recorta o rosto (com padding) do bitmap original
     * 2. Rotaciona o recorte em torno do seu próprio centro
     * 3. Redimensiona para 112×112
     *
     * IMPORTANTE: a versão anterior fazia o inverso (rotacionava o bitmap inteiro
     * e depois aplicava as coordenadas originais), o que deslocava o rosto para fora
     * do crop e resultava em capturas de fundo em vez de rosto.
     */
    private fun alignFace(
        bitmap: Bitmap,
        leftEyeX: Float, leftEyeY: Float,
        rightEyeX: Float, rightEyeY: Float,
        bounds: android.graphics.Rect
    ): Bitmap {
        // Passo 1: recorta o rosto com padding (já calculado em bounds)
        val left   = maxOf(0, bounds.left)
        val top    = maxOf(0, bounds.top)
        val right  = minOf(bitmap.width, bounds.right)
        val bottom = minOf(bitmap.height, bounds.bottom)

        if (right <= left || bottom <= top) {
            return Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        }

        val cropped = Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top)

        // Passo 2: calcula o ângulo de inclinação dos olhos e rotaciona o recorte
        // em torno do seu centro (não do bitmap inteiro — erro da versão anterior)
        val angle = Math.toDegrees(
            atan2((rightEyeY - leftEyeY).toDouble(), (rightEyeX - leftEyeX).toDouble())
        ).toFloat()

        val matrix = Matrix()
        matrix.postRotate(-angle, cropped.width / 2f, cropped.height / 2f)

        val rotated = Bitmap.createBitmap(cropped, 0, 0, cropped.width, cropped.height, matrix, true)

        // Passo 3: redimensiona para o tamanho de entrada do modelo
        return Bitmap.createScaledBitmap(rotated, INPUT_SIZE, INPUT_SIZE, true)
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

                    // Filtra apenas rostos com probabilidade de olhos abertos acima de 0.2
                    // Rostos de bonecas/mascotes/pôsteres geralmente têm probabilidade nula ou muito baixa
                    // O threshold 0.2 é intencional baixo para não rejeitar fotos com olhos semi-cerrados
                    val humanFaces = faces.filter { f ->
                        val leftProb  = f.leftEyeOpenProbability  ?: 0f
                        val rightProb = f.rightEyeOpenProbability ?: 0f
                        leftProb > 0.2f || rightProb > 0.2f
                    }

                    if (humanFaces.isEmpty()) {
                        promise.reject("NO_FACE", "Nenhum rosto humano detectado na imagem (possível boneca ou mascote)")
                        return@addOnSuccessListener
                    }

                    // Seleciona o rosto com maior pontuação combinada de área × confiança dos olhos.
                    // Isso evita que mascotes/banners (que têm área grande mas olhos pouco confiáveis)
                    // sejam preferidos em relação ao rosto humano real na foto.
                    val face = humanFaces.maxByOrNull { f ->
                        val area = f.boundingBox.width().toFloat() * f.boundingBox.height()
                        // Usa o MAX dos olhos em vez da média: evita penalizar demais rostos
                        // com um olho semi-fechado (ex: pai olhando de lado, olho parcialmente tapado).
                        // Com a média, um olho com prob 0.02 derrubava o score inteiro.
                        val eyeScore = maxOf(f.leftEyeOpenProbability ?: 0f, f.rightEyeOpenProbability ?: 0f)
                        area * eyeScore
                    }!!

                    // Expande o bounding box em 30% para cada lado.
                    // Modelos ArcFace/FaceNet foram treinados com imagens que incluem
                    // testa, queixo e orelhas — sem esse contexto, o embedding piora muito.
                    val rawBounds = face.boundingBox
                    val padX = (rawBounds.width()  * 0.3f).toInt()
                    val padY = (rawBounds.height() * 0.3f).toInt()
                    val bounds = android.graphics.Rect(
                        maxOf(0, rawBounds.left   - padX),
                        maxOf(0, rawBounds.top    - padY),
                        minOf(bitmap.width,  rawBounds.right  + padX),
                        minOf(bitmap.height, rawBounds.bottom + padY)
                    )

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

                    // DEBUG: salva todos os rostos detectados + o rosto selecionado final
                    try {
                        val cacheDir = reactApplicationContext.externalCacheDir ?: reactApplicationContext.cacheDir
                        val slot = if (java.io.File(cacheDir, "slot1_selected.jpg").exists()) "2" else "1"

                        // Salva cada rosto detectado individualmente
                        faces.forEachIndexed { idx, f ->
                            val b = f.boundingBox
                            val l = maxOf(0, b.left); val t = maxOf(0, b.top)
                            val r = minOf(bitmap.width, b.right); val bo = minOf(bitmap.height, b.bottom)
                            if (r > l && bo > t) {
                                val crop = Bitmap.createBitmap(bitmap, l, t, r - l, bo - t)
                                val lEye = f.leftEyeOpenProbability ?: -1f
                                val rEye = f.rightEyeOpenProbability ?: -1f
                                val area = b.width() * b.height()
                                // Nome: slot_candidato_INDEX_area_AREA_eyes_L_R.jpg
                                val name = "slot${slot}_candidate${idx}_area${area}_L${String.format("%.2f", lEye)}_R${String.format("%.2f", rEye)}.jpg"
                                java.io.FileOutputStream(java.io.File(cacheDir, name)).use { out ->
                                    crop.compress(android.graphics.Bitmap.CompressFormat.JPEG, 90, out)
                                }
                            }
                        }

                        // Salva o rosto final que foi ao modelo
                        java.io.FileOutputStream(java.io.File(cacheDir, "slot${slot}_selected.jpg")).use { out ->
                            faceBitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 95, out)
                        }
                    } catch (e: Exception) { /* ignora erros de debug */ }

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
     * Prepara a imagem e roda o modelo ONNX via ONNX Runtime.
     *
     * Diferença importante em relação ao TFLite:
     * - TFLite esperava formato NHWC: [batch, altura, largura, canais]
     * - ONNX Runtime espera formato NCHW: [batch, canais, altura, largura]
     *
     * Isso significa que os pixels precisam ser agrupados por canal (R, G, B)
     * em vez de por pixel (R G B | R G B | R G B ...).
     *
     * A normalização (pixel - 127.5) / 128.0 continua a mesma — é o padrão ArcFace.
     */
    private fun runFaceNet(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // Formato NCHW com ordem BGR.
        // Testes empíricos mostraram que BGR produz embeddings melhores com o w600k_r50,
        // possivelmente porque o modelo ONNX foi exportado sem a conversão BGR→RGB do blobFromImages.
        // Android Bitmap armazena ARGB: bits 23-16 = R, 15-8 = G, 7-0 = B.
        val floatArray = FloatArray(3 * INPUT_SIZE * INPUT_SIZE)
        val planeSize = INPUT_SIZE * INPUT_SIZE

        for (i in pixels.indices) {
            val pixel = pixels[i]
            floatArray[i]                 = ((pixel        and 0xFF).toFloat() - 127.5f) / 128.0f  // B (canal 0)
            floatArray[planeSize + i]     = ((pixel shr 8  and 0xFF).toFloat() - 127.5f) / 128.0f  // G (canal 1)
            floatArray[2 * planeSize + i] = ((pixel shr 16 and 0xFF).toFloat() - 127.5f) / 128.0f  // R (canal 2)
        }

        val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        val tensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(floatArray), shape)

        // Obtém o nome da entrada dinamicamente (evita hardcode que varia por modelo)
        val inputName = ortSession!!.inputNames.iterator().next()
        val results = ortSession!!.run(mapOf(inputName to tensor))

        return (results[0].value as Array<FloatArray>)[0]
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
