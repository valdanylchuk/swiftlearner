#[derive(Debug, Clone, PartialEq)]
pub struct Perceptron {
    weights: Vec<f32>,
    bias: f32,
}

impl Perceptron {
    pub fn new(weights: Vec<f32>, bias: f32) -> Self { Self { weights, bias } }

    pub fn with_size(input_size: usize, initial_value: f32) -> Self {
        Self { weights: vec![initial_value; input_size], bias: initial_value }
    }

    pub fn activate(&self, input: &[f32]) -> bool {
        if input.len() != self.weights.len() {
            panic!("Input size mismatch: expected {}, got {}", self.weights.len(), input.len());
        }

        let mut sum = self.bias;
        for i in 0..input.len() {
            sum += self.weights[i] * input[i];
        }
        sum > 0.0
    }

    pub fn learn(&mut self, input: &[f32], target: bool) {
        if input.len() != self.weights.len() {
            panic!("Input size mismatch");
        }

        let error = (if target { 1.0 } else { 0.0 })
                  - (if self.activate(input) { 1.0 } else { 0.0 });

        self.bias += error;
        for i in 0..self.weights.len() {
            self.weights[i] += error * input[i];
        }
    }

    pub fn train(&mut self, examples: &[(Vec<f32>, bool)], epochs: usize) {
        for _ in 0..epochs {
            for (input, target) in examples {
                self.learn(input, *target);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_function() {
        let p = Perceptron::new(vec![2.0, 3.0], -25.0);
        assert!(!p.activate(&[4.0, 5.0]));
        assert!(p.activate(&[4.0, 6.0]));
    }

    #[test]
    fn test_learn_and_function() {
        let and_examples = vec![
            (vec![0.0, 0.0], false),
            (vec![0.0, 1.0], false),
            (vec![1.0, 0.0], false),
            (vec![1.0, 1.0], true),
        ];

        let mut p = Perceptron::with_size(2, 0.0);
        p.train(&and_examples, 10);
        for (input, target) in &and_examples {
            assert_eq!(p.activate(input), *target);
        }
    }

    #[test]
    fn test_learn_or_function() {
        let or_examples = vec![
            (vec![0.0, 0.0], false),
            (vec![0.0, 1.0], true),
            (vec![1.0, 0.0], true),
            (vec![1.0, 1.0], true),
        ];

        let mut p = Perceptron::with_size(2, 0.0);
        p.train(&or_examples, 10);
        for (input, target) in &or_examples {
            assert_eq!(p.activate(input), *target);
        }
    }

    #[test]
    #[should_panic(expected = "Input size mismatch: expected 2, got 3")]
    fn test_input_size_mismatch() {
        let p = Perceptron::with_size(2, 0.0);
        p.activate(&[1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "Input size mismatch")]
    fn test_input_size_mismatch_learn() {
        let mut p = Perceptron::with_size(2, 0.0);
        p.learn(&[1.0, 2.0, 3.0], true);
    }
}