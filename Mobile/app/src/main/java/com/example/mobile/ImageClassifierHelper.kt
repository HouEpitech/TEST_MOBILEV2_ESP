package com.example.mobile

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.task.vision.classifier.ImageClassifier
import java.io.IOException

class ImageClassifierHelper(context: Context) {
    private var imageClassifier: ImageClassifier

    init {
        try {
            val options = ImageClassifier.ImageClassifierOptions.builder()
                .setMaxResults(1)
                .build()

            imageClassifier = ImageClassifier.createFromFileAndOptions(
                context, MODEL_FILE, options
            )
        } catch (e: IOException) {
            Log.e("TFLite", "Erreur d'initialisation du modèle", e)
            throw e
        }
    }

    fun classify(bitmap: Bitmap): String {
        // Conversion et prétraitement
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        val image = org.tensorflow.lite.support.image.TensorImage.fromBitmap(resizedBitmap)

        // Inférence
        val results = imageClassifier.classify(image)

        // Récupération des résultats
        return results.firstOrNull()?.categories?.firstOrNull()?.let { category ->
            "${category.label} (\"${category.score.times(100.0f).format(2)}%\")"
        } ?: "Aucune classification détectée"
    }

    companion object {
        private const val MODEL_FILE = "model.tflite"
        private const val IMAGE_SIZE = 224
    }
}

// Extension pour formater les Float
fun Float.format(decimals: Int) = "%.${decimals}f".format(this.toDouble())
