package com.agrivision.data.local

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface DiagnosticDao {
    @Insert
    suspend fun insertScan(diagnostic: DiagnosticEntity)

    @Query("SELECT * FROM diagnostic_history ORDER BY timestamp DESC")
    fun getAllScans(): Flow<List<DiagnosticEntity>>
}
