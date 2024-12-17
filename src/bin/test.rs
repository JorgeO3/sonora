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

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    let mut counter = 0;

    while let Ok(packet) = wave.next_packet() {
        if let AudioBufferRef::S16(buf) = decoder.decode(&packet)? {
            let left_sample = buf.chan(0).first();
            let right_sample = buf.chan(1).first();
            println!("Left: {:?}, Right: {:?}", left_sample, right_sample);

            if counter == 180 {
                break;
            }

            counter += 1;

            // raw_samples.extend(buf.chan(0).iter().chain(buf.chan(1).iter()));
        }
    }

    Ok(())
}
