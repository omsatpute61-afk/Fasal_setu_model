package com.agrivision.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.agrivision.ml.DiagnosticResult
import com.agrivision.ui.components.AudioPlayerFAB

@Composable
fun ResultTabsScreen(
    result: DiagnosticResult,
    cropName: String,
    onScanAnother: () -> Unit
) {
    var selectedTab by remember { mutableStateOf(0) }
    val tabs = listOf("Overview", "Disease", "Pest", "Treatment")

    Column(modifier = Modifier.fillMaxSize().background(Color(0xFFF5F5F5))) {
        
        // 1. TOP BAR (Crop & Health Score)
        Surface(
            color = Color(0xFF2E7D32),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(text = "Target: $cropName", color = Color.White, fontSize = 18.sp)
                Text(
                    text = "Health Score: ${result.healthIndex.toInt()}/100", 
                    color = Color.White, 
                    fontSize = 24.sp, 
                    fontWeight = FontWeight.Bold
                )
            }
        }

        // 2. TAB SELECTION BAR
        TabRow(
            selectedTabIndex = selectedTab,
            containerColor = Color.White,
            contentColor = Color(0xFF2E7D32)
        ) {
            tabs.forEachIndexed { index, title ->
                Tab(
                    selected = selectedTab == index,
                    onClick = { selectedTab = index },
                    text = { Text(title, fontWeight = FontWeight.Bold) }
                )
            }
        }

        // 3. TAB CONTENT
        Box(modifier = Modifier.weight(1f).padding(16.dp)) {
            when (selectedTab) {
                0 -> OverviewTab(result)
                1 -> DiseaseTab(result)
                2 -> PestTab(result)
                3 -> TreatmentTab(result)
            }
        }

        // 4. BOTTOM ACTION
        Button(
            onClick = onScanAnother,
            modifier = Modifier.fillMaxWidth().padding(16.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2E7D32))
        ) {
            Text("Scan Another Plant", fontSize = 18.sp)
        }
    }
}

@Composable
fun OverviewTab(result: DiagnosticResult) {
    Column {
        Text("Primary Urgency", fontWeight = FontWeight.Bold, fontSize = 20.sp)
        Text(result.urgency, color = if (result.urgency == "Critical") Color.Red else Color.Green, fontSize = 18.sp)
        Spacer(modifier = Modifier.height(16.dp))
        
        result.gatekeeperWarning?.let {
            Card(colors = CardDefaults.cardColors(containerColor = Color(0xFFFFF3E0))) {
                Text(it, modifier = Modifier.padding(16.dp), color = Color(0xFFE65100))
            }
        }
    }
}


@Composable
fun DiseaseTab(result: DiagnosticResult) {
    Column {
        Text("Detected Pathogen", fontWeight = FontWeight.Bold, fontSize = 20.sp)
        Text(result.diseaseScientificName ?: "None", fontStyle = androidx.compose.ui.text.font.FontStyle.Italic, fontSize = 18.sp)
        Spacer(modifier = Modifier.height(8.dp))
        Text("Affected Area: ${result.diseaseAffectedArea}%")
        
        Spacer(modifier = Modifier.weight(1f))
        AudioPlayerFAB("Disease Detected: ${result.diseaseScientificName ?: "None"}. Affected area is ${result.diseaseAffectedArea} percent.")
    }
}

@Composable
fun PestTab(result: DiagnosticResult) {
    Column {
        Text("Detected Pest", fontWeight = FontWeight.Bold, fontSize = 20.sp)
        Text(result.primaryPest ?: "None", fontStyle = androidx.compose.ui.text.font.FontStyle.Italic, fontSize = 18.sp)
        Spacer(modifier = Modifier.height(8.dp))
        Text("Total Insect Count: ${result.insectCount}")
        if (result.insectCount > 5) {
            Text("WARNING: Economic Threshold Level Exceeded!", color = Color.Red, fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
fun TreatmentTab(result: DiagnosticResult) {
    Column {
        Card(
            colors = CardDefaults.cardColors(containerColor = Color(0xFFE8F5E9)),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("🌱 Organic Control", fontWeight = FontWeight.Bold, color = Color(0xFF1B5E20))
                Text(result.organicAdvisory)
            }
        }
        Spacer(modifier = Modifier.height(16.dp))
        Card(
            colors = CardDefaults.cardColors(containerColor = Color(0xFFFFEBEE)),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("🧪 Chemical Control", fontWeight = FontWeight.Bold, color = Color(0xFFB71C1C))
                Text(result.chemicalAdvisory)
            }
        }
        
        Spacer(modifier = Modifier.weight(1f))
        AudioPlayerFAB("Organic Control: ${result.organicAdvisory}. Chemical Control: ${result.chemicalAdvisory}")
    }
}
