use {
    rayon::prelude::*,
    rustfft::{num_complex::Complex, FftPlanner},
    std::{
        fs::File,
        io::{BufWriter, Write},
    },
    symphonia::{
        core::{
            audio::{AudioBufferRef, Signal},
            codecs::DecoderOptions,
            formats::FormatReader,
            io::MediaSourceStream,
        },
        default::formats::WavReader as SymphoniaWavReader,
    },
};

const INPUT_FILE: &str = "data/big_input.wav";
const OUTPUT_FILE: &str = "output.txt";
const CHUNK_SIZE: usize = 1024 * 4;
const FUZ_FACTOR: usize = 2;
const MIN_FREQ: usize = 40;
const MAX_FREQ: usize = 300;

fn hash(p: &[usize; 301]) -> usize {
    let p1 = p[40] / FUZ_FACTOR;
    let p2 = p[80] / FUZ_FACTOR;
    let p3 = p[120] / FUZ_FACTOR;
    let p4 = p[180] / FUZ_FACTOR;
    (p4 * 100_000_000) + (p3 * 100_000) + (p2 * 100) + p1
}

fn get_index(x: usize) -> usize {
    match x {
        0..=40 => 40,
        41..=80 => 80,
        81..=120 => 120,
        121..=180 => 180,
        _ => 300,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Inicializar cronómetro
    let time = std::time::Instant::now();

    // Abrir archivo de entrada
    let src = File::open(INPUT_FILE)?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    let mut wave = SymphoniaWavReader::try_new(mss, &Default::default())?;
    let track = wave
        .default_track()
        .ok_or("No se encontró el track de audio")?;
    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    // Leer y decodificar, almacenando en buffer
    let mut raw_samples: Vec<i16> = Vec::with_capacity(1024 * 1024); // Ajustar capacidad según necesidad

    while let Ok(packet) = wave.next_packet() {
        if let AudioBufferRef::S16(buf) = decoder.decode(&packet)? {
            raw_samples.extend(buf.chan(0).iter().chain(buf.chan(1).iter()));
        }
    }

    // Convertir a números complejos en paralelo
    let mut freqs: Vec<Complex<f32>> = raw_samples
        .par_iter()
        .map(|&x| Complex::new(x as f32, 0.0))
        .collect();

    // Alinear y rellenar con ceros
    let new_len = freqs.len().div_ceil(CHUNK_SIZE) * CHUNK_SIZE;
    freqs.resize(new_len, Complex::new(0.0, 0.0));

    println!("Time reading and decoding: {:?}", time.elapsed());

    // Realizar FFT
    let fft_start = std::time::Instant::now();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(CHUNK_SIZE);

    freqs.par_chunks_mut(CHUNK_SIZE).for_each(|chunk| {
        let mut buffer = chunk.to_vec();
        fft.process(&mut buffer);
        // Copiar resultados de vuelta si es necesario
        for (i, sample) in buffer.iter().enumerate() {
            chunk[i] = *sample;
        }
    });

    println!("Tiempo de FFT: {:?}", fft_start.elapsed());

    // Preparar índices de frecuencia
    let hash_start = std::time::Instant::now();
    let freq_indexes: Vec<usize> = (MIN_FREQ..MAX_FREQ).map(get_index).collect();

    // Realizar hashing en paralelo
    let results: Vec<usize> = freqs
        .par_chunks(CHUNK_SIZE)
        .map(|chunk| {
            let mut points = [0_usize; 301];
            let mut hscores = [0.0_f32; 301];
            for (freq, &index) in (MIN_FREQ..MAX_FREQ).zip(freq_indexes.iter()) {
                if freq >= CHUNK_SIZE {
                    continue; // Evitar out-of-bounds
                }
                let sample = chunk[freq];
                let mag = sample.re * sample.re + sample.im * sample.im;
                if mag > hscores[index] {
                    points[index] = freq;
                    hscores[index] = mag;
                }
            }
            hash(&points)
        })
        .collect();

    // Escribir resultados
    let file = File::create(OUTPUT_FILE)?;
    let mut buf = BufWriter::with_capacity(1024 * 1024, file);
    for result in results {
        writeln!(buf, "{}", result)?;
    }
    buf.flush()?;

    println!("Tiempo de hashing: {:?}", hash_start.elapsed());

    Ok(())
}
