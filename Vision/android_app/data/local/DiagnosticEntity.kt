package com.agrivision.data.local

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "diagnostic_history")
data class DiagnosticEntity(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val timestamp: Long,
    val cropName: String,
    val diseaseScientificName: String?,
    val insectCount: Int,
    val healthIndex: Float,
    val imagePath: String,
    val isSynced: Boolean = false
)
