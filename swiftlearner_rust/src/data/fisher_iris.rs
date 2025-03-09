// src/fisher_iris.rs
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub type VecF = Vec<f32>;
pub type DataEntry = (usize, VecF); // Class index and parameters
pub type DataSet = Vec<DataEntry>;

pub fn load_data(
    filename: &str,
    random_seed: Option<u64>,
) -> Result<(DataSet, DataSet), Box<dyn Error>> {
    let path = Path::new(filename);
    let file = File::open(&path)?;
    let reader = BufReader::new(file);
    let mut data: DataSet = Vec::new();

    // Skip the header line.  Handle errors gracefully.
    let mut lines = reader.lines();
    lines.next().ok_or("File is empty")??; // Skip header and handle potential error

    for line_result in lines {
        let line = line_result?;
        let mut values = line.split(',');

        // Extract itemClass. Handle parse errors.
        let item_class: usize = values
            .next()
            .ok_or("Missing class value")?
            .parse()
            .map_err(|_| "Invalid class value")?;

        // Extract parameters. Handle parse errors.
        let mut parameters: VecF = Vec::new();
        for value in values {
            parameters.push(value.parse().map_err(|_| "Invalid parameter value")?);
        }

        data.push((item_class, parameters));
    }

    // Shuffle the data using the provided seed or a default seed.
    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    };
    data.shuffle(&mut rng);

    // Split into training and test sets (2/3 training, 1/3 test)
    let training_size = data.len() * 2 / 3;
    let training_set = data[..training_size].to_vec();
    let test_set = data[training_size..].to_vec();

    Ok((training_set, test_set))
}