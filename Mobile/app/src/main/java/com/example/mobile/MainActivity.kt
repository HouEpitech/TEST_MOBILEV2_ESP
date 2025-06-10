package com.example.mobile
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.mobile.R
import java.io.IOException



class MainActivity : AppCompatActivity() {
    private lateinit var classifier: ImageClassifierHelper
    private lateinit var imageView: ImageView
    private lateinit var resultText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        try {
            classifier = ImageClassifierHelper(applicationContext)
        } catch (e: IOException) {
            Log.e("TFLite", "Erreur d'initialisation du modèle", e)
        }

        imageView = findViewById(R.id.image_view)
        resultText.text = findViewById(R.id.result_text)

        findViewById<Button>(R.id.select_button).setOnClickListener { pickImage() }
    }

    private fun pickImage() {
        val intent = Intent(Intent.ACTION_PICK).apply {
            type = "image/*"
        }
        startActivityForResult(intent, 1)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK && data != null) {
            data.data?.let { uri ->
                val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)
                imageView.setImageBitmap(bitmap)
                resultText.text = classifier.classify(bitmap)
            }
        }
    }
}
