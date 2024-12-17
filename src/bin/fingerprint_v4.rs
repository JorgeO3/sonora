use crossbeam::channel::{bounded, Receiver, Sender};
use mimalloc::MiMalloc;
use rustfft::{num_complex::Complex, FftPlanner};
use std::{
    fs::File,
    io::{BufWriter, Write},
    sync::Arc,
    time::Instant,
};
use symphonia::{
    core::{
        audio::{AudioBufferRef, Signal},
        codecs::DecoderOptions,
        formats::FormatReader,
        io::MediaSourceStream,
    },
    default::{formats::WavReader as SymphoniaWavReader, get_codecs},
};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const INPUT_FILE: &str = "big_input.wav";
const OUTPUT_FILE: &str = "output.txt";
const CHUNK_SIZE: usize = 1024 * 4;
const FUZ_FACTOR: usize = 2;
const MIN_FREQ: usize = 40;
const MAX_FREQ: usize = 300;

#[inline]
fn hash(p: &[usize; 301]) -> usize {
    let p1 = p[40] / FUZ_FACTOR;
    let p2 = p[80] / FUZ_FACTOR;
    let p3 = p[120] / FUZ_FACTOR;
    let p4 = p[180] / FUZ_FACTOR;
    (p4 * 100_000_000) + (p3 * 100_000) + (p2 * 100) + p1
}

#[inline]
fn get_index(x: usize) -> usize {
    match x {
        0..=40 => 40,
        41..=80 => 80,
        81..=120 => 120,
        121..=180 => 180,
        _ => 300,
    }
}

fn decode_audio(sender: Sender<Vec<i16>>) -> Result<(), Box<dyn std::error::Error>> {
    let src = File::open(INPUT_FILE)?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    let mut wave = SymphoniaWavReader::try_new(mss, &Default::default())?;
    let track = wave
        .default_track()
        .ok_or("No se encontró el track de audio")?;
    let mut decoder = get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut raw_samples = Vec::with_capacity(CHUNK_SIZE * 10);

    while let Ok(packet) = wave.next_packet() {
        if let Ok(AudioBufferRef::S16(buf)) = decoder.decode(&packet) {
            raw_samples.extend(buf.chan(0).iter().chain(buf.chan(1).iter()));
            while raw_samples.len() >= CHUNK_SIZE {
                let chunk: Vec<i16> = raw_samples.drain(0..CHUNK_SIZE).collect();
                sender.send(chunk)?;
            }
        }
    }

    if !raw_samples.is_empty() {
        sender.send(raw_samples)?;
    }

    Ok(())
}

fn process_audio(receiver: Receiver<Vec<i16>>) -> Result<(), Box<dyn std::error::Error>> {
    let mut planner = FftPlanner::new();
    let fft = Arc::new(planner.plan_fft_forward(CHUNK_SIZE));

    let freq_indexes = (MIN_FREQ..MAX_FREQ).map(get_index).collect::<Vec<usize>>();

    let file = File::create(OUTPUT_FILE)?;
    let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, file);

    let mut freqs = vec![Complex::default(); CHUNK_SIZE];
    let mut points = [0usize; 301];
    let mut hscores = [0.0f32; 301];

    for raw_chunk in receiver.iter() {
        freqs.clear();
        freqs.extend(
            raw_chunk
                .iter()
                .map(|&sample| Complex::new(sample as f32, 0.0)),
        );
        freqs.resize(CHUNK_SIZE, Complex::default());

        fft.process(&mut freqs);

        points.fill(0);
        hscores.fill(0.0);

        for (freq, &index) in (MIN_FREQ..MAX_FREQ).zip(freq_indexes.iter()) {
            if freq >= CHUNK_SIZE {
                continue;
            }
            let sample = freqs[freq];
            let mag = sample.norm_sqr();
            if mag > hscores[index] {
                points[index] = freq;
                hscores[index] = mag;
            }
        }

        writeln!(writer, "{}", hash(&points))?;
    }

    writer.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    let (sender, receiver) = bounded(20);

    let producer_handle = std::thread::spawn(move || {
        if let Err(e) = decode_audio(sender) {
            eprintln!("Error en el hilo productor: {}", e);
        }
    });

    let consumer_handle = std::thread::spawn(move || {
        if let Err(e) = process_audio(receiver) {
            eprintln!("Error en el hilo consumidor: {}", e);
        }
    });

    producer_handle.join().expect("El hilo productor falló");
    consumer_handle.join().expect("El hilo consumidor falló");

    println!(
        "Tiempo de decodificación y procesamiento conjunto: {:?}",
        start_time.elapsed()
    );

    Ok(())
}
