//! The samples of a NIfTI start at `vox_offset`, not where the header parser stopped.
//!
//! The files under `resources/vox_offset` hold one 4x5x6 float32 ramp, sample (x, y, z) being
//! x + 4 y + 20 z: a single file with 48 bytes of padding and no extension (`vox_offset` 400), a
//! single file with one 48-byte extension (`vox_offset` 400), and a pair whose header carries one
//! extension and whose image starts with 16 bytes of padding (`vox_offset` 16).

extern crate nifti;

use nifti::{
    InMemNiftiVolume, NiftiObject, NiftiVolume, RandomAccessNiftiVolume, ReaderOptions,
    ReaderStreamedOptions,
};

fn assert_ramp(volume: &InMemNiftiVolume) {
    assert_eq!(volume.dim(), &[4, 5, 6]);
    for z in 0..6u16 {
        for y in 0..5u16 {
            for x in 0..4u16 {
                let value = volume.get_f32(&[x, y, z]).unwrap();
                assert_eq!(value, f32::from(x + 4 * y + 20 * z), "at ({x}, {y}, {z})");
            }
        }
    }
}

#[test]
fn a_single_file_padded_before_its_samples_reads_them_at_vox_offset() {
    let obj = ReaderOptions::new()
        .read_file("resources/vox_offset/padded.nii")
        .unwrap();
    assert_eq!(obj.header().vox_offset, 400.0);
    assert!(obj.extensions().is_empty());
    assert_ramp(obj.volume());
}

#[test]
fn a_single_file_with_an_extension_still_reads_its_samples_after_it() {
    let obj = ReaderOptions::new()
        .read_file("resources/vox_offset/extension.nii")
        .unwrap();
    assert_eq!(obj.extensions().len(), 1);
    assert_ramp(obj.volume());
}

#[test]
fn a_streamed_single_file_reads_its_samples_at_vox_offset() {
    let obj = ReaderStreamedOptions::new()
        .read_file("resources/vox_offset/padded.nii")
        .unwrap();
    assert_eq!(obj.volume().dim(), &[4, 5, 6]);
    for (z, slice) in obj.into_volume().enumerate() {
        let slice = slice.unwrap();
        for y in 0..5u16 {
            for x in 0..4u16 {
                let value = slice.get_f32(&[x, y]).unwrap();
                assert_eq!(
                    value,
                    f32::from(x + 4 * y + 20 * z as u16),
                    "at ({x}, {y}, {z})"
                );
            }
        }
    }
}

#[test]
fn a_pair_reads_the_header_extensions_and_the_image_at_vox_offset() {
    let obj = ReaderOptions::new()
        .read_file("resources/vox_offset/pair.hdr")
        .unwrap();
    assert_eq!(obj.header().vox_offset, 16.0);
    assert_eq!(obj.extensions().len(), 1);
    assert_eq!(obj.extensions().iter().next().unwrap().code(), 6);
    assert_ramp(obj.volume());

    let obj = ReaderOptions::new()
        .read_file_pair(
            "resources/vox_offset/pair.hdr",
            "resources/vox_offset/pair.img",
        )
        .unwrap();
    assert_eq!(obj.extensions().len(), 1);
    assert_ramp(obj.volume());
}

#[test]
fn a_pair_header_cut_inside_an_extension_is_refused() {
    let dir = std::env::temp_dir().join(format!("nifti-rs-cut-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let hdr = std::fs::read("resources/vox_offset/pair.hdr").unwrap();
    std::fs::write(dir.join("cut.hdr"), &hdr[..hdr.len() - 8]).unwrap();
    std::fs::copy("resources/vox_offset/pair.img", dir.join("cut.img")).unwrap();
    let result = ReaderOptions::new().read_file(dir.join("cut.hdr"));
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(result.is_err(), "an extension cut short is not a shorter extension");
}
