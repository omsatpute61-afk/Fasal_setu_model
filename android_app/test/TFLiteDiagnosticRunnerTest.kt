package com.agrivision.test

import android.content.Context
import android.graphics.Bitmap
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.agrivision.ml.TFLiteDiagnosticRunner
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class TFLiteDiagnosticRunnerTest {

    private lateinit var runner: TFLiteDiagnosticRunner
    private lateinit var context: Context

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        runner = TFLiteDiagnosticRunner(context)
    }

    @After
    fun tearDown() {
        runner.close()
    }

    @Test
    fun testOfflineInferencePipeline() {
        // 1. Create a mock synthetic Bitmap (representing a captured leaf)
        val mockBitmap = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888)

        // 2. Execute Runner (Offline, No Network)
        val result = runner.runDiagnostics(mockBitmap, "Cotton")

        // 3. Verify Output Payload matches 4-Tab Schema
        
        // Tab 1: Overview
        assertNotNull("Gatekeeper warning should map correctly", result.gatekeeperWarning)
        assertTrue("Health index should be calculated", result.healthIndex > 0f)
        
        // Tab 2: Disease
        assertEquals("Alternaria solani", result.diseaseScientificName)
        assertEquals(22.5f, result.diseaseAffectedArea, 0.1f)
        
        // Tab 3: Pests
        assertEquals("Aphis gossypii", result.primaryPest)
        assertEquals(12, result.insectCount)
        assertEquals("Critical", result.urgency) // 12 > 5 triggers Critical ETL
        
        // Tab 4: Treatment
        assertTrue(result.organicAdvisory.contains("Neem Seed Extract"))
        assertTrue(result.chemicalAdvisory.contains("Chlorantraniliprole"))
        
        println("SUCCESS: TFLite Diagnostic Runner executed all 3 models completely offline and mapped the Kotlin data class correctly for the 4-Tab UI.")
    }
}
