package com.agrivision.worker

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.agrivision.data.local.AppDatabase

class SyncWorker(
    appContext: Context,
    workerParams: WorkerParameters
) : CoroutineWorker(appContext, workerParams) {

    override suspend fun doWork(): Result {
        val database = AppDatabase.getDatabase(applicationContext)
        val dao = database.diagnosticDao()

        return try {
            val unsyncedScans = dao.getUnsyncedScans()
            if (unsyncedScans.isNotEmpty()) {
                // Mock Network Call
                // val response = apiService.uploadScans(unsyncedScans)
                kotlinx.coroutines.delay(1000) // Simulate network delay
                
                // On Success, mark them as synced
                dao.markAsSynced(unsyncedScans.map { it.id })
            }
            Result.success()
        } catch (e: Exception) {
            e.printStackTrace()
            Result.retry()
        }
    }
}
