use rand::Rng;
use rand_mt::Mt;
use std::cell::RefCell;
use std::f32::NEG_INFINITY;

pub type VecF = Vec<f32>;

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    weights: VecF,
    output: f32,
    error: f32,
}

// Static functions for f and ff (outside impl Node)
fn f(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn ff(old_output: f32) -> f32 {
    old_output * (1.0 - old_output)
}

impl Node {
    fn new(w: VecF) -> Self {
        Node {
            weights: w,
            output: 0.0,
            error: 0.0,
        }
    }

    fn calculate_output_for(&self, input: &[f32]) -> f32 {
        f(input.iter().zip(self.weights.iter()).map(|(x, w)| x * w).sum())
    }

    fn update(&mut self, old_inputs: &[f32], partial_error: f32, rate: f32) {
        self.error = ff(self.output) * partial_error;
        for (i, &old_input) in old_inputs.iter().enumerate() {
            self.weights[i] += rate * self.error * old_input;
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BackpropNet {
    pub hidden_layer: Vec<Node>,
    pub output_layer: Vec<Node>,
    hidden_layer_output: VecF,
}

impl BackpropNet {
    pub fn new(hidden_layer: Vec<Node>, output_layer: Vec<Node>) -> Self {
        let hidden_layer_output = vec![0.0; hidden_layer.len()];
        BackpropNet {
            hidden_layer,
            output_layer,
            hidden_layer_output,
        }
    }

    /// Return Index of the highest output
    pub fn predict(&mut self, input: &[f32]) -> usize {
        for (i, node) in self.hidden_layer.iter().enumerate() {
            self.hidden_layer_output[i] = node.calculate_output_for(input);
        }

        let mut max_output = NEG_INFINITY;
        let mut max_output_index = 0;

        for (i, node) in self.output_layer.iter().enumerate() {
            let node_output = node.calculate_output_for(&self.hidden_layer_output);
            if node_output > max_output {
                max_output = node_output;
                max_output_index = i;
            }
        }
        max_output_index
    }

    /// Calculate output[0], for testing only
    pub fn test_out(&mut self, input: &[f32]) -> f32 {
        for (i, node) in self.hidden_layer.iter().enumerate() {
            self.hidden_layer_output[i] = node.calculate_output_for(input);
        }
        self.output_layer[0].calculate_output_for(&self.hidden_layer_output)
    }

    /// Learn an example using backpropagation.
    pub fn learn(&mut self, example: &Example, rate: f32) -> &mut Self {
        if rate > 1.0 || rate <= 0.0 {
            panic!("learning rate must be between 0 and 1");
        }

        // Forward pass
        for (i, node) in self.hidden_layer.iter_mut().enumerate() {
            node.output = node.calculate_output_for(&example.input);
            self.hidden_layer_output[i] = node.output;
        }

        // Forward pass for output layer
        for node in self.output_layer.iter_mut() {
            node.output = node.calculate_output_for(&self.hidden_layer_output);
        }

        // Backward pass: Update output layer
        for (j, node) in self.output_layer.iter_mut().enumerate() {
            let partial_error = example.target[j] - node.output;
            node.update(&self.hidden_layer_output, partial_error, rate);
        }

        // Backward pass: Update hidden layer
        for (i, node) in self.hidden_layer.iter_mut().enumerate() {
            let mut partial_error = 0.0;
            for output_node in self.output_layer.iter() {
                partial_error += output_node.weights[i] * output_node.error;
            }
            node.update(&example.input, partial_error, rate);
        }

        self
    }

    pub fn train(&mut self, examples: &[Example], epochs: usize, rate: f32) -> &mut Self {
        for _ in 0..epochs {
            for example in examples {
                self.learn(example, rate);
            }
        }
        self
    }

    pub fn random_net(n_input: usize, n_hidden: usize, n_output: usize, seed: Option<u32>) -> Self {
        let generator = RefCell::new(match seed {
            Some(s) => Mt::new(s),
            None => Mt::new_unseeded(),
        });

        let rand_weights = move |n| {
            let mut gen = generator.borrow_mut();
            (0..n).map(|_| gen.random_range(0.0..1.0)).collect::<VecF>()
        };

        let hidden_layer = (0..n_hidden)
            .map(|_| Node::new(rand_weights(n_input)))
            .collect();
        let output_layer = (0..n_output)
            .map(|_| Node::new(rand_weights(n_hidden)))
            .collect();
        BackpropNet::new(hidden_layer, output_layer)
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    pub input: VecF,
    pub target: VecF,
}


#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn reproduce_known_example() {
        let ex = Example {
            input: vec![0.35, 0.9],
            target: vec![0.5],
        };

        let hidden_layer = vec![Node::new(vec![0.1, 0.8]), Node::new(vec![0.4, 0.6])];
        let output_layer = vec![Node::new(vec![0.3, 0.9])];

        let mut nn = BackpropNet::new(hidden_layer, output_layer);
        let learned = nn.learn(&ex, 1.0);

        assert_approx_eq!(learned.hidden_layer[0].output, 0.6803, 0.0001);
        assert_approx_eq!(learned.hidden_layer[1].output, 0.6637, 0.0001);
        assert_approx_eq!(learned.output_layer[0].output, 0.6903, 0.0001);

        assert_approx_eq!(learned.output_layer[0].error, -0.0406, 0.0001);
        assert_approx_eq!(learned.output_layer[0].weights[0], 0.2723, 0.0001);
        assert_approx_eq!(learned.output_layer[0].weights[1], 0.8730, 0.0001);

        assert_approx_eq!(learned.hidden_layer[0].error, -0.0025, 0.0002);
        assert_approx_eq!(learned.hidden_layer[0].weights[0], 0.0991, 0.0002);
        assert_approx_eq!(learned.hidden_layer[0].weights[1], 0.7977, 0.0002);

        assert_approx_eq!(learned.hidden_layer[1].error, -0.008, 0.0002);
        assert_approx_eq!(learned.hidden_layer[1].weights[0], 0.3972, 0.0002);
        assert_approx_eq!(learned.hidden_layer[1].weights[1], 0.5927, 0.0002);

        let out = learned.test_out(&ex.input);
        assert_approx_eq!(out, 0.682, 0.0001);
    }

    #[test]
    fn learn_or_function() {
        let or_examples = vec![
            Example {
                input: vec![0.0, 0.0],
                target: vec![0.0],
            },
            Example {
                input: vec![0.0, 1.0],
                target: vec![1.0],
            },
            Example {
                input: vec![1.0, 0.0],
                target: vec![1.0],
            },
            Example {
                input: vec![1.0, 1.0],
                target: vec![1.0],
            },
        ];

        let mut nn = BackpropNet::random_net(2, 2, 1, Some(10));
        nn.train(&or_examples, 1000, 1.0);

        for ex in &or_examples {
            let out = nn.test_out(&ex.input);
            assert_approx_eq!(out, ex.target[0], 0.1);
        }
    }

    #[test]
    fn learn_and_function() {
        let and_examples = vec![
            Example {
                input: vec![0.0, 0.0],
                target: vec![0.0],
            },
            Example {
                input: vec![0.0, 1.0],
                target: vec![0.0],
            },
            Example {
                input: vec![1.0, 0.0],
                target: vec![0.0],
            },
            Example {
                input: vec![1.0, 1.0],
                target: vec![1.0],
            },
        ];

        let mut nn = BackpropNet::random_net(2, 3, 1, Some(10));
        nn.train(&and_examples, 1000, 1.0);

        for ex in &and_examples {
            let out = nn.test_out(&ex.input);
            assert_approx_eq!(out, ex.target[0], 0.1);
        }
    }

    #[test]
    fn learn_xor_function() {
        let xor_examples = vec![
            Example {
                input: vec![0.0, 0.0],
                target: vec![0.0],
            },
            Example {
                input: vec![0.0, 1.0],
                target: vec![1.0],
            },
            Example {
                input: vec![1.0, 0.0],
                target: vec![1.0],
            },
            Example {
                input: vec![1.0, 1.0],
                target: vec![0.0],
            },
        ];

        let mut nn = BackpropNet::random_net(2, 4, 1, Some(10));
        nn.train(&xor_examples, 6000, 1.0);

        for ex in &xor_examples {
            let out = nn.test_out(&ex.input);
            assert_approx_eq!(out, ex.target[0], 0.1);
        }
    }
}