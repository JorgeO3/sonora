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

const INPUT_FILE: &str = "data/dulce_carita.wav";
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

    let time = std::time::Instant::now();
    while let Ok(packet) = wave.next_packet() {
        if let AudioBufferRef::S16(buf) = decoder.decode(&packet)? {
            raw_samples.extend(buf.chan(0).iter().chain(buf.chan(1).iter()));
        }
    }
    println!("Tiempo de lectura y decodificación: {:?}", time.elapsed());

    Ok(())
}
