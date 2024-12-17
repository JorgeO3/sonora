use std::fs::File;

use symphonia::{
    core::{
        audio::{AudioBufferRef, Signal},
        codecs::DecoderOptions,
        formats::FormatReader,
        io::MediaSourceStream,
    },
    default::formats::WavReader as SymphoniaWavReader,
};

const INPUT_FILE: &str = "data/big_input.wav";
const OUTPUT_FILE: &str = "output.txt";
const CHUNK_SIZE: usize = 1024 * 4;
const FUZ_FACTOR: usize = 2;
const MIN_FREQ: usize = 40;
const MAX_FREQ: usize = 300;

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

    std::hint::black_box(|| {
        while let Ok(packet) = wave.next_packet() {
            if let AudioBufferRef::S16(data) = decoder.decode(&packet).unwrap() {
                let mut data2 = data.to_owned();
                let data23 = data2.to_mut();
            }
        }
    })();

    println!("Elapsed time: {:?}", time.elapsed());

    Ok(())
}
