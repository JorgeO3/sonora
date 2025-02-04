use {
    rayon::prelude::*,
    rustfft::{num_complex::Complex, FftPlanner},
    std::{
        arch::x86_64::*,
        fs::File,
        io::{BufWriter, Write},
        ops::Rem,
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

const INPUT_FILE: &str = "big_input.wav";
const OUTPUT_FILE: &str = "output.txt";
const CHUNK_SIZE: usize = 1024 * 4;
const FUZ_FACTOR: usize = 2;
const MIN_FREQ: usize = 40;
const MAX_FREQ: usize = 300;

fn hash(p: &[usize; 301]) -> usize {
    let p1 = p[40];
    let p2 = p[80];
    let p3 = p[120];
    let p4 = p[180];
    (p4 - p4.rem(FUZ_FACTOR)) * 100_000_000
        + (p3 - p3.rem(FUZ_FACTOR)) * 100_000
        + (p2 - p2.rem(FUZ_FACTOR)) * 100
        + (p1 - p1.rem(FUZ_FACTOR))
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
    let time = std::time::Instant::now();
    let src = File::open(INPUT_FILE)?;
    let file_size = src.metadata()?.len() as usize;

    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    let mut wave = SymphoniaWavReader::try_new(mss, &Default::default())?;
    let track = wave.default_track().unwrap();
    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    // Read and decode, storing iterators
    let estimated_samples = file_size / 2;
    let mut raw_samples = Vec::with_capacity(estimated_samples);

    // Read and decode in a single thread, but collect raw samples
    while let Ok(packet) = wave.next_packet() {
        if let Ok(AudioBufferRef::S16(buf)) = decoder.decode(&packet) {
            raw_samples.extend(buf.chan(0).iter().chain(buf.chan(1)).cloned());
        }
    }

    // Convert to complex numbers in parallel
    let mut freqs: Vec<Complex<f32>> = raw_samples
        .par_iter()
        .map(|&x| Complex::new(x as f32, 0.0))
        .collect();

    let new_len = freqs.len().div_ceil(CHUNK_SIZE) * CHUNK_SIZE;
    freqs.resize(new_len, Complex::default());
    println!("Time reading and decoding: {:?}", time.elapsed());

    let time = std::time::Instant::now();
    // Perform FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(CHUNK_SIZE);
    freqs
        .par_chunks_mut(CHUNK_SIZE)
        .for_each(|chunk| fft.process(chunk));
    println!("Time fft: {:?}", time.elapsed());

    let time = std::time::Instant::now();
    let freq_indexes: Vec<(usize, usize)> =
        (MIN_FREQ..MAX_FREQ).map(|x| (x, get_index(x))).collect();

    let results: Vec<usize> = freqs
        .par_chunks(CHUNK_SIZE)
        .map(|chunk| {
            let mut points = [0; MAX_FREQ + 1];
            let mut hscores = [0.0; MAX_FREQ + 1];
            for (freq, index) in freq_indexes.iter() {
                let mag = unsafe {
                    let real = _mm_loadu_ps(&chunk[*freq].re as *const f32);
                    let imag = _mm_loadu_ps(&chunk[*freq].im as *const f32);
                    let mag =
                        _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(real, real), _mm_mul_ps(imag, imag)));
                    _mm_cvtss_f32(mag)
                };
                if mag > hscores[*index] {
                    points[*index] = *freq;
                    hscores[*index] = mag;
                }
            }
            hash(&points)
        })
        .collect();

    let file = File::create(OUTPUT_FILE)?;
    let mut buf = BufWriter::with_capacity(1024 * 1024 * 100, file);
    results.iter().for_each(|result| {
        writeln!(buf, "{}", result).unwrap();
    });

    println!("Time hashing: {:?}", time.elapsed());
    buf.flush()?;
    Ok(())
}
