package com.agrivision.ui.components

import android.content.Context
import android.speech.tts.TextToSpeech
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import java.util.Locale

@Composable
fun AudioPlayerFAB(textToRead: String) {
    val context = LocalContext.current
    var isTtsReady by remember { mutableStateOf(false) }
    var selectedLanguage by remember { mutableStateOf(Locale.US) }
    var expanded by remember { mutableStateOf(false) }

    val tts = remember {
        TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                isTtsReady = true
            }
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            tts.stop()
            tts.shutdown()
        }
    }

    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(8.dp)) {
        // Play Button
        FloatingActionButton(
            onClick = {
                if (isTtsReady) {
                    tts.language = selectedLanguage
                    tts.speak(textToRead, TextToSpeech.QUEUE_FLUSH, null, null)
                }
            },
            containerColor = MaterialTheme.colorScheme.primaryContainer,
            contentColor = MaterialTheme.colorScheme.onPrimaryContainer
        ) {
            Text("🔊 Play", modifier = Modifier.padding(horizontal = 12.dp))
        }

        // Language Dropdown
        Box(modifier = Modifier.padding(start = 8.dp)) {
            Button(onClick = { expanded = true }) {
                Text(selectedLanguage.displayLanguage)
            }
            DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                DropdownMenuItem(
                    text = { Text("English") },
                    onClick = {
                        selectedLanguage = Locale.US
                        expanded = false
                    }
                )
                DropdownMenuItem(
                    text = { Text("Hindi") },
                    onClick = {
                        // Create Hindi Locale
                        selectedLanguage = Locale("hi", "IN")
                        expanded = false
                    }
                )
            }
        }
    }
}
