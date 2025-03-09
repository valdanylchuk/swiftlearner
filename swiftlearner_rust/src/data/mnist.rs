// src/mnist.rs
use flate2::read::GzDecoder;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::error::Error;
use crate::data::fisher_iris::{DataSet, VecF};


pub const IMAGE_WIDTH: usize = 28;
pub const IMAGE_HEIGHT: usize = 28;
pub const IMAGE_SIZE: usize = IMAGE_WIDTH * IMAGE_HEIGHT;
pub const TRAIN_SET_SIZE: usize = 60000;
pub const TEST_SET_SIZE: usize = 10000;

pub fn load_data(
    random_seed: Option<u32>,
    n_train_points: usize,
    train_images_file: &str,
    train_labels_file: &str,
    test_images_file: &str,
    test_labels_file: &str,
) -> Result<(DataSet, DataSet), Box<dyn Error>> {

    let train_images_gz = File::open(Path::new(train_images_file))?;
    let train_labels_gz = File::open(Path::new(train_labels_file))?;
    let test_images_gz = File::open(Path::new(test_images_file))?;
    let test_labels_gz = File::open(Path::new(test_labels_file))?;

    let mut train_images_decoder = GzDecoder::new(train_images_gz);
    let mut train_labels_decoder = GzDecoder::new(train_labels_gz);
    let mut test_images_decoder = GzDecoder::new(test_images_gz);
    let mut test_labels_decoder = GzDecoder::new(test_labels_gz);

    let train_labels = read_labels(&mut train_labels_decoder, n_train_points)?;
    let test_labels = read_labels(&mut test_labels_decoder, TEST_SET_SIZE)?;
    let train_images = read_images(&mut train_images_decoder, n_train_points)?;
    let test_images = read_images(&mut test_images_decoder, TEST_SET_SIZE)?;


    let mut training_set: DataSet = Vec::with_capacity(n_train_points);
    let mut test_set: DataSet = Vec::with_capacity(TEST_SET_SIZE);


    for i in 0..n_train_points {
        training_set.push((train_labels[i] as usize, train_images[i].clone()));
    }
    for i in 0..TEST_SET_SIZE {
        test_set.push((test_labels[i] as usize, test_images[i].clone()));
    }

    // Shuffle the training data
    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed as u64),
        None => StdRng::from_os_rng(),
    };
    training_set.shuffle(&mut rng);

    Ok((training_set, test_set))
}

fn read_labels<R: Read>(file: &mut R, set_size: usize) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut labels = vec![0u8; set_size];

    // Skip the magic number and number of items (4 bytes each) by reading and discarding
    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;

    file.read_exact(&mut labels)?;
    Ok(labels)
}

fn read_images<R: Read>(file: &mut R, set_size: usize) -> Result<Vec<VecF>, Box<dyn Error>>{
    let mut images = vec![vec![0.0; IMAGE_SIZE]; set_size];

    // Skip the magic number, number of images, rows, and columns (4 bytes each)
    let mut header = [0u8; 16];
    file.read_exact(&mut header)?;


    for i in 0..set_size {
        let mut buffer = [0u8; IMAGE_SIZE];
        file.read_exact(&mut buffer)?;
        for (j, &pixel) in buffer.iter().enumerate() {
            images[i][j] = pixel as f32;
        }
    }

    Ok(images)
}

// Unit Tests for MNIST data loading
#[cfg(test)]
mod tests {
    use super::*;
    const TRAIN_IMAGES_FILE: &str = "resources/mnist/train-images-idx3-ubyte.gz";
    const TRAIN_LABELS_FILE: &str = "resources/mnist/train-labels-idx1-ubyte.gz";
    const TEST_IMAGES_FILE: &str = "resources/mnist/t10k-images-idx3-ubyte.gz";
    const TEST_LABELS_FILE: &str = "resources/mnist/t10k-labels-idx1-ubyte.gz";


    #[test]
    fn test_dataset_size() -> Result<(), Box<dyn std::error::Error>>{
        let (training_set, test_set) = load_data(Some(0), 1000,
        TRAIN_IMAGES_FILE,
        TRAIN_LABELS_FILE,
        TEST_IMAGES_FILE,
        TEST_LABELS_FILE
        )?;
        assert_eq!(training_set.len(), 1000);
        assert_eq!(test_set.len(), TEST_SET_SIZE);
        Ok(())
    }

    #[test]
    fn test_image_size() -> Result<(), Box<dyn std::error::Error>> {
        let (training_set, _) = load_data(Some(0), 1,
        TRAIN_IMAGES_FILE,
        TRAIN_LABELS_FILE,
        TEST_IMAGES_FILE,
        TEST_LABELS_FILE
        )?;
        let image = &training_set[0].1;
        assert_eq!(image.len(), IMAGE_SIZE);
        Ok(())
    }
    #[test]
    fn test_read_labels_file() -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(TRAIN_LABELS_FILE)?;
        let mut decoder = GzDecoder::new(file);
        let labels = read_labels(&mut decoder, 10)?;
        assert_eq!(labels.len(), 10);
        Ok(())
    }

    #[test]
    fn test_read_images_file() -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(TRAIN_IMAGES_FILE)?;
        let mut decoder = GzDecoder::new(file);
        let images = read_images(&mut decoder, 10)?;
        assert_eq!(images.len(), 10);
        assert_eq!(images[0].len(), IMAGE_SIZE);
        Ok(())
    }
}