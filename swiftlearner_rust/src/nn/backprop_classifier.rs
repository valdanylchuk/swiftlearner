// src/backprop_classifier.rs

use crate::nn::backprop_net::{BackpropNet, Example, VecF};
use crate::data::fisher_iris::DataSet;

pub struct BackpropClassifier {
    n_params: usize,
    normalized_input: VecF,
    learned: BackpropNet,
    normalize_func: Box<dyn Fn(f32) -> f32 + 'static>, // Keep the boxed closure
}

impl BackpropClassifier {
    pub fn new( // No generic type parameter
        training_set: DataSet,
        n_hidden: usize,
        n_times: usize,
        learn_rate: f32,
        normalize: Box<dyn Fn(f32) -> f32>,
        seed: Option<u32>,
    ) -> Self
    {
        let n_classes = training_set.iter().map(|(c, _)| *c).max().unwrap_or(0) + 1;
        let n_params = if training_set.is_empty() {
            0
        } else {
            training_set[0].1.len()
        };
        let normalized_input = vec![0.0; n_params];
		// The only place where 'normalize' is actually used (not just stored as a field):
        let examples = Self::create_examples(&training_set, &normalize, n_classes);

        let mut learned = BackpropNet::random_net(n_params, n_hidden, n_classes, seed);
        learned.train(&examples, n_times, learn_rate);

        BackpropClassifier {
            n_params,
            normalized_input,
            learned,
            normalize_func: normalize, // Directly store the boxed closure
        }
    }

	// Still need to take a reference to the Box here:
    pub fn predict(&mut self, parameters: &VecF) -> usize {
        if parameters.len() != self.n_params {
            panic!("Wrong number of parameters");
        }

        for (i, &param) in parameters.iter().enumerate() {
            self.normalized_input[i] = (self.normalize_func)(param); // Call the boxed closure
        }

        self.learned.predict(&self.normalized_input)
    }

	// Still need to take a reference to the Box here:
    fn create_examples(
        training_set: &DataSet,
        normalize: &Box<dyn Fn(f32) -> f32>, // Take a reference to the boxed closure
        n_classes: usize,
    ) -> Vec<Example>
    {
        let mut examples = Vec::with_capacity(training_set.len());

        for &(class_idx, ref params) in training_set {
            let mut target_one_hot = vec![0.0; n_classes];
            target_one_hot[class_idx] = 1.0;

            let normalized_params: VecF = params.iter().map(|&x| normalize(x)).collect();
            examples.push(Example {
                input: normalized_params,
                target: target_one_hot,
            });
        }
        examples
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::fisher_iris::load_data as load_iris_data;
    use crate::data::mnist::load_data as load_mnist_data;
    use crate::data::mnist;
    use std::time::Instant;

    #[test]
    fn test_fisher_iris_classification() -> Result<(), Box<dyn std::error::Error>> {
        let (training_set, test_set) = load_iris_data("resources/FisherIris.csv", None)?;
        let normalize = |x: f32| (x - 25.0) / 25.0;
		// Need to Box::new() when we pass it in:
        let mut classifier = BackpropClassifier::new(training_set, 3, 5000, 1.0, Box::new(normalize), None);
        let mut correct_predictions = 0;
        for (species, params) in &test_set {
            if classifier.predict(params) == *species {
                correct_predictions += 1;
            }
        }
        let accuracy = correct_predictions as f64 / test_set.len() as f64;
        assert!(accuracy > 0.8, "Accuracy below expected threshold");
        Ok(())
    }
    #[test]
    fn test_empty_training_set() {
        let training_set = vec![];
        let normalize = |x: f32| (x - 25.0) / 25.0;
        let _classifier = BackpropClassifier::new(training_set, 3, 5000, 1.0, Box::new(normalize), None);
    }

    #[test]
    #[should_panic(expected = "Wrong number of parameters")]
    fn test_incorrect_parameter_count() {
        let (training_set, _) =
            load_iris_data("resources/FisherIris.csv", None).expect("Failed to load data");
        let normalize = |x: f32| (x - 25.0) / 25.0;
        let mut classifier = BackpropClassifier::new(training_set, 3, 5000, 1.0, Box::new(normalize), None);
        let incorrect_params = vec![1.0, 2.0];
        classifier.predict(&incorrect_params);
    }

    #[test]
    fn test_classify_mnist() -> Result<(), Box<dyn std::error::Error>> {
        let n_hidden = 70;
        let n_repeat = 1;
        let learn_rate = 1.0;
        let expected_accuracy = 0.87;
        let seed: Option<u32> = Some(0);

        let (training_set, test_set) = load_mnist_data(
            seed,
            mnist::TRAIN_SET_SIZE,
            "resources/mnist/train-images-idx3-ubyte.gz",
            "resources/mnist/train-labels-idx1-ubyte.gz",
            "resources/mnist/t10k-images-idx3-ubyte.gz",
            "resources/mnist/t10k-labels-idx1-ubyte.gz",
        )?;

        let scale = |x: f32| x / 255.0;

        let mut mean = 0.0;
        let mut num_samples = 0;
         for (_, image) in training_set.iter().take(10000) {
             for &pixel_value in image {
                mean += scale(pixel_value) as f64;
                num_samples += 1;
            }
        }

        mean /= num_samples as f64;
        println!("mean: {}", mean);

        // Use `move` to capture by value:
        let normalize = move |x: f32| scale(x) - (mean as f32);

        println!("creating the classifier");
        let start = Instant::now();

		// Need to Box::new() when we pass it in:
        let mut classifier = BackpropClassifier::new(
            training_set,
            n_hidden,
            n_repeat,
            learn_rate,
            Box::new(normalize), // Box the closure here
            seed
        );

        println!("checking the accuracy");
        let mut correct_predictions = 0;
        for (digit, params) in &test_set {
            if classifier.predict(params) == *digit {
                correct_predictions += 1;
            }
        }
        let accuracy = correct_predictions as f64 / test_set.len() as f64;

        let duration = start.elapsed();

        println!("MNIST backprop time: {:.2?}s", duration);
        println!("Accuracy: {}", accuracy);

        assert!(accuracy > expected_accuracy, "Accuracy below expectation");
        Ok(())
    }
}