package com.agrivision.ml

import android.content.Context
import android.graphics.Bitmap
import com.agrivision.data.local.AppDatabase
import com.agrivision.data.local.DiagnosticEntity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil

data class DiagnosticResult(
    val healthIndex: Float,
    val urgency: String,
    val gatekeeperWarning: String?,
    val diseaseScientificName: String?,
    val diseaseAffectedArea: Float,
    val primaryPest: String?,
    val insectCount: Int,
    val organicAdvisory: String,
    val chemicalAdvisory: String
)

class TFLiteDiagnosticRunner(context: Context) {
    
    private var gatekeeperInterpreter: Interpreter? = null
    private var diseaseInterpreter: Interpreter? = null
    private var pestInterpreter: Interpreter? = null

    init {
        // Configure options for blazing fast Edge CPU inference
        val options = Interpreter.Options().apply {
            setNumThreads(4)
            // Enforce XNNPACK delegate for highly optimized CPU processing
            setUseXNNPACK(true) 
        }

        try {
            gatekeeperInterpreter = Interpreter(FileUtil.loadMappedFile(context, "gatekeeper_int8.tflite"), options)
            diseaseInterpreter = Interpreter(FileUtil.loadMappedFile(context, "disease_edge_int8.tflite"), options)
            pestInterpreter = Interpreter(FileUtil.loadMappedFile(context, "pest_edge_int8.tflite"), options)
        } catch (e: Exception) {
            e.printStackTrace()
            // Graceful fallback if models aren't in assets yet during dev
        }
    }

    fun runDiagnostics(bitmap: Bitmap, crop: String, imagePath: String = "", db: AppDatabase? = null): DiagnosticResult {
        // 1. GATEKEEPER
        val gatekeeperScore = runGatekeeper(bitmap)
        val warning = if (gatekeeperScore in 0.35f..0.55f) "Partial foliage. Scanning with adapted sensitivity." else null
        
        if (gatekeeperScore < 0.35f) {
            // (Skipping DB write for rejected scans)
            // ...
        }

        // 2. DISEASE DETECTION
        val (diseaseName, area) = runDiseaseModel(bitmap)

        // 3. PEST DETECTION
        val (pestName, count) = runPestModel(bitmap)
        
        // 4. REGISTRY AGGREGATION & SCHEMATIZATION
        val healthIndex = maxOf(0f, 100f - (area * 2f) - (count * 1.5f))
        val urgency = if (count > 5 || area > 20f) "Critical" else "Normal"
        
        // PHASE 13: MOCK WEATHER INTEGRATION
        // Note: In production, run this in a coroutine prior to this synchronous method, or make this method suspend.
        // For the offline edge runner, we mock a synchronous weather fetch if cached.
        val mockWeatherWarning = "⚠️ Rain Predicted: Do not apply chemical sprays today to prevent chemical wash-off."
        
        val baseChemicalAdvisory = "Apply Chlorantraniliprole for $pestName if ETL exceeded."
        val finalChemicalAdvisory = "$baseChemicalAdvisory\n\n$mockWeatherWarning"
        
        // 5. ASYNC OFFLINE DB WRITE (PHASE 12)
        db?.let {
            CoroutineScope(Dispatchers.IO).launch {
                it.diagnosticDao().insertScan(
                    DiagnosticEntity(
                        timestamp = System.currentTimeMillis(),
                        cropName = crop,
                        diseaseScientificName = diseaseName,
                        insectCount = count,
                        healthIndex = healthIndex,
                        imagePath = imagePath
                    )
                )
            }
        }
        
        // Return 4-Tab matched schema
        return DiagnosticResult(
            healthIndex = healthIndex,
            urgency = urgency,
            gatekeeperWarning = warning,
            diseaseScientificName = diseaseName,
            diseaseAffectedArea = area,
            primaryPest = pestName,
            insectCount = count,
            organicAdvisory = "Apply 5% Neem Seed Extract for $pestName. Prune lower leaves for $diseaseName.",
            chemicalAdvisory = finalChemicalAdvisory
        )
    }

    private fun runGatekeeper(bitmap: Bitmap): Float {
        // Mock inference parsing logic for TFLite bytebuffers
        // return outputBuffer.getFloat(0)
        return 0.45f // Triggering Soft Pass
    }

    private fun runDiseaseModel(bitmap: Bitmap): Pair<String, Float> {
        // Parse bounding boxes and classes from TFLite YOLO-seg output
        return Pair("Alternaria solani", 22.5f)
    }

    private fun runPestModel(bitmap: Bitmap): Pair<String, Int> {
        // Parse YOLO TFLite output
        return Pair("Aphis gossypii", 12)
    }
    
    fun close() {
        gatekeeperInterpreter?.close()
        diseaseInterpreter?.close()
        pestInterpreter?.close()
    }
}
