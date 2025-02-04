use std::error::Error;
use std::f32::consts::PI;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use sha1::{Digest, Sha1};

/// Estructura para almacenar el espectrograma.
struct Spectrogram {
    frequencies: Vec<f32>,
    times: Vec<f32>,
    magnitudes: Vec<Vec<f32>>,
}

/// Estructura para representar un pico detectado.
#[derive(Debug, Clone)]
struct Peak {
    time: f32,
    frequency: f32,
}

/// Estructura para almacenar un hash y su tiempo de ocurrencia.
struct HashEntry {
    hash: String,
    time: f32,
}

/// Carga un archivo de audio WAV y devuelve una señal mono y normalizada.
///
/// # Argumentos
///
/// * `file_path` - Ruta al archivo de audio.
///
/// # Retorna
///
/// * `Result<Vec<f32>, String>` - Vector de muestras de audio normalizadas o un mensaje de error.
fn load_audio(file_path: &str) -> Result<Vec<f32>, String> {
    // Abre el archivo WAV.
    let mut reader = hound::WavReader::open(file_path)
        .map_err(|e| format!("Error abriendo archivo WAV: {}", e))?;

    // Obtiene las especificaciones del WAV.
    let spec = reader.spec();

    // Asegura que el audio sea de 16 bits por muestra y PCM.
    if spec.bits_per_sample != 16 || spec.sample_format != hound::SampleFormat::Int {
        return Err("Solo se soportan archivos WAV de 16 bits y formato PCM.".to_string());
    }

    // Lee todas las muestras.
    let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap_or(0)).collect();

    // Convierte a mono si es necesario.
    let num_channels = spec.channels as usize;
    let mut mono_samples = Vec::new();

    if num_channels == 1 {
        mono_samples = samples.iter().map(|&s| s as f32).collect();
    } else {
        for frame in samples.chunks(num_channels) {
            let sum: f32 = frame.iter().map(|&s| s as f32).sum();
            mono_samples.push(sum / num_channels as f32);
        }
    }

    // Encuentra el máximo absoluto para normalización.
    let max_amplitude = mono_samples.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);

    if max_amplitude == 0.0 {
        return Err("El archivo de audio está completamente silencioso.".to_string());
    }

    // Normaliza las muestras.
    let normalized_samples: Vec<f32> = mono_samples.iter().map(|&s| s / max_amplitude).collect();

    Ok(normalized_samples)
}

/// Genera una ventana de Hann.
///
/// # Argumentos
///
/// * `size` - Tamaño de la ventana.
///
/// # Retorna
///
/// * `Vec<f32>` - Ventana de Hann.
fn hann_window(size: usize) -> Vec<f32> {
    let pi = PI;
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * pi * i as f32 / size as f32).cos()))
        .collect()
}

/// Calcula el espectrograma utilizando FFT.
///
/// # Argumentos
///
/// * `samples` - Vector de muestras de audio.
/// * `sample_rate` - Tasa de muestreo.
/// * `window_size` - Tamaño de la ventana para FFT.
/// * `overlap` - Solapamiento entre ventanas.
///
/// # Retorna
///
/// * `Spectrogram` - Espectrograma calculado.
fn calculate_spectrogram(
    samples: &[f32],
    sample_rate: usize,
    window_size: usize,
    overlap: usize,
) -> Spectrogram {
    let hop_size = window_size - overlap;
    let num_windows = if samples.len() < window_size {
        0
    } else {
        ((samples.len() - window_size) / hop_size) + 1
    };

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(window_size);
    let window = hann_window(window_size);

    let mut magnitudes = Vec::with_capacity(num_windows);
    let mut frequencies = Vec::new();
    let mut times = Vec::new();

    for i in 0..num_windows {
        let start = i * hop_size;
        let end = start + window_size;
        let windowed: Vec<Complex<f32>> = samples[start..end]
            .iter()
            .zip(window.iter())
            .map(|(s, w)| Complex::new(*s * w, 0.0))
            .collect();

        let mut buffer = windowed.clone();
        fft.process(&mut buffer);

        // Calcula magnitudes.
        let magnitude: Vec<f32> = buffer
            .iter()
            .take(window_size / 2)
            .map(|c| c.norm())
            .collect();
        magnitudes.push(magnitude);

        // Solo calcular frecuencias y tiempos una vez.
        if frequencies.is_empty() {
            let freq_res = sample_rate as f32 / window_size as f32;
            frequencies = (0..(window_size / 2))
                .map(|i| i as f32 * freq_res)
                .collect();
        }

        let time = start as f32 / sample_rate as f32;
        times.push(time);
    }

    Spectrogram {
        frequencies,
        times,
        magnitudes,
    }
}

/// Encuentra picos en el espectrograma.
///
/// # Argumentos
///
/// * `spectrogram` - Espectrograma calculado.
/// * `amp_min` - Umbral mínimo de amplitud para detectar picos.
/// * `neighborhood_size` - Tamaño del vecindario para la detección de máximos locales.
///
/// # Retorna
///
/// * `Vec<Peak>` - Vector de picos detectados.
fn find_peaks(spectrogram: &Spectrogram, amp_min: f32, neighborhood_size: usize) -> Vec<Peak> {
    let mut peaks = Vec::new();
    let num_freqs = spectrogram.frequencies.len();
    let num_times = spectrogram.times.len();

    for t in 0..num_times {
        for f in 0..num_freqs {
            let magnitude = spectrogram.magnitudes[t][f];
            if magnitude < amp_min {
                continue;
            }

            let mut is_peak = true;

            // Define el rango del vecindario.
            let f_start = f.saturating_sub(neighborhood_size);

            let f_end = if f + neighborhood_size < num_freqs {
                f + neighborhood_size
            } else {
                num_freqs - 1
            };
            let t_start = t.saturating_sub(neighborhood_size);
            let t_end = if t + neighborhood_size < num_times {
                t + neighborhood_size
            } else {
                num_times - 1
            };

            // Verifica si es un pico local.
            'check: for tt in t_start..=t_end {
                for ff in f_start..=f_end {
                    if spectrogram.magnitudes[tt][ff] > magnitude {
                        is_peak = false;
                        break 'check;
                    }
                }
            }

            if is_peak {
                peaks.push(Peak {
                    time: spectrogram.times[t],
                    frequency: spectrogram.frequencies[f],
                });
            }
        }
    }

    peaks
}

/// Genera hashes únicos a partir de los picos detectados.
///
/// # Argumentos
///
/// * `peaks` - Vector de picos detectados.
/// * `fan_value` - Número de picos a emparejar con cada pico actual.
/// * `max_delta_t` - Máximo intervalo de tiempo en segundos para emparejar picos.
///
/// # Retorna
///
/// * `Vec<HashEntry>` - Vector de hashes generados.
fn generate_hashes(peaks: &[Peak], fan_value: usize, max_delta_t: f32) -> Vec<HashEntry> {
    let mut hashes = Vec::new();
    let mut peaks_sorted = peaks.to_vec();

    // Ordenar los picos por tiempo.
    peaks_sorted.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

    for i in 0..peaks_sorted.len() {
        let current_peak = &peaks_sorted[i];
        for j in 1..=fan_value {
            if i + j >= peaks_sorted.len() {
                break;
            }
            let paired_peak = &peaks_sorted[i + j];
            let delta_t = paired_peak.time - current_peak.time;
            if delta_t > max_delta_t {
                break;
            }

            // Crear una cadena única para el par de picos.
            let hash_input = format!(
                "{}|{}|{}",
                current_peak.frequency.round() as u32,
                paired_peak.frequency.round() as u32,
                delta_t.round() as u32
            );

            // Generar el hash utilizando SHA-1 y tomar los primeros 20 caracteres hexadecimales.
            let mut hasher = Sha1::new();
            hasher.update(hash_input.as_bytes());
            let hash_result = hasher.finalize();
            let hash_hex = hex::encode(&hash_result[..10]); // 10 bytes = 20 caracteres hex

            hashes.push(HashEntry {
                hash: hash_hex,
                time: current_peak.time,
            });
        }
    }

    hashes
}

fn main() -> Result<(), Box<dyn Error>> {
    // Parámetros
    let audio_file = "big_input.wav"; // Reemplaza con la ruta de tu archivo de audio WAV
    let window_size = 4096;
    let overlap = 2048;
    let amp_min = 10.0;
    let neighborhood_size = 20;
    let fan_value = 15;
    let max_delta_t = 5.0;

    println!("Cargando y preprocesando el audio...");
    // Cargar y preprocesar el audio
    let samples = load_audio(audio_file)?;
    println!(
        "Audio cargado y normalizado. Cantidad de muestras: {}",
        samples.len()
    );

    println!("Calculando el espectrograma...");
    // Calcular el espectrograma
    let spectrogram = calculate_spectrogram(&samples, 44100, window_size, overlap);
    println!(
        "Espectrograma calculado. Frecuencias: {}, Tiempos: {}",
        spectrogram.frequencies.len(),
        spectrogram.times.len()
    );

    println!("Detectando picos en el espectrograma...");
    // Encontrar picos en el espectrograma
    let peaks = find_peaks(&spectrogram, amp_min, neighborhood_size);
    println!("Cantidad de picos detectados: {}", peaks.len());

    println!("Generando hashes a partir de los picos...");
    // Generar hashes a partir de los picos
    let hashes = generate_hashes(&peaks, fan_value, max_delta_t);
    println!("Cantidad de hashes generados: {}", hashes.len());

    // Mostrar algunos hashes generados
    println!("\nAlgunos hashes generados:");
    for (i, hash_entry) in hashes.iter().take(10).enumerate() {
        println!(
            "Hash {}: {} en el tiempo {:.2} segundos",
            i + 1,
            hash_entry.hash,
            hash_entry.time
        );
    }

    Ok(())
}
