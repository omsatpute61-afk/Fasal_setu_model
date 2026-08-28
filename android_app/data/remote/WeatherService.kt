package com.agrivision.data.remote

// In production, configure Retrofit using: Retrofit.Builder().baseUrl("https://api.openweathermap.org/").build()

data class WeatherForecast(
    val precipitationMm: Float,
    val windSpeedKmh: Float
)

class WeatherService {
    
    // Mocking the Retrofit suspend call
    suspend fun getLocalWeatherForecast(lat: Double, lon: Double): WeatherForecast {
        // Simulating network delay
        kotlinx.coroutines.delay(500)
        
        // Mock data: Randomize weather for demo purposes
        val isRaining = Math.random() > 0.7
        val isWindy = Math.random() > 0.8
        
        return WeatherForecast(
            precipitationMm = if (isRaining) 5.2f else 0.0f,
            windSpeedKmh = if (isWindy) 22.0f else 8.5f
        )
    }
    
    fun getSprayAdvisoryWarning(forecast: WeatherForecast): String? {
        val warnings = mutableListOf<String>()
        
        if (forecast.precipitationMm > 2.5f) {
            warnings.add("⚠️ Rain Predicted: Do not apply chemical sprays today to prevent chemical wash-off.")
        }
        
        if (forecast.windSpeedKmh > 15.0f) {
            warnings.add("⚠️ High Wind: Avoid spraying to prevent chemical drift.")
        }
        
        return if (warnings.isNotEmpty()) warnings.joinToString("\n") else null
    }
}
