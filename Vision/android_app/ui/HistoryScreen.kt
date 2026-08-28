package com.agrivision.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.agrivision.data.local.DiagnosticEntity
import kotlinx.coroutines.flow.Flow
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun HistoryScreen(historyFlow: Flow<List<DiagnosticEntity>>) {
    val history by historyFlow.collectAsState(initial = emptyList())

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Text("Offline Scan History", fontSize = 24.sp, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(16.dp))

        if (history.isEmpty()) {
            Text("No scans found. Start by scanning a crop!")
        } else {
            LazyColumn {
                items(history) { scan ->
                    HistoryCard(scan)
                }
            }
        }
    }
}

@Composable
fun HistoryCard(scan: DiagnosticEntity) {
    val dateFormat = SimpleDateFormat("MMM dd, yyyy - HH:mm", Locale.getDefault())
    val dateString = dateFormat.format(Date(scan.timestamp))

    Card(
        modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(dateString, color = Color.Gray, fontSize = 12.sp)
            Spacer(modifier = Modifier.height(4.dp))
            
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(scan.cropName, fontWeight = FontWeight.Bold, fontSize = 18.sp)
                Text("Health: ${scan.healthIndex.toInt()}", fontWeight = FontWeight.Bold, color = if(scan.healthIndex > 75) Color(0xFF2E7D32) else Color.Red)
            }
            
            Spacer(modifier = Modifier.height(8.dp))
            Text("Disease: ${scan.diseaseScientificName ?: "None"}")
            Text("Pests Found: ${scan.insectCount}")
            
            // Note: In production, we would use Coil to load scan.imagePath here
            Text("Image Saved: ${scan.imagePath}", fontSize = 10.sp, color = Color.LightGray)
        }
    }
}
