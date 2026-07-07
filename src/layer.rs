use crate::math::{sigmoid, sigmoid_derivate};
use rand::RngExt;

#[derive(Clone, Debug)]
pub struct Layer<const IN_SIZE: usize, const OUT_SIZE: usize> {
    pub weights: [[f64; IN_SIZE]; OUT_SIZE],
    pub biases: [f64; OUT_SIZE],
}

impl<const IN_SIZE: usize, const OUT_SIZE: usize> Default for Layer<IN_SIZE, OUT_SIZE> {
    fn default() -> Self {
        let mut rng = rand::rng();
        let limit = (6.0 / (IN_SIZE + OUT_SIZE) as f64).sqrt();

        Self {
            weights: std::array::from_fn(|_| {
                std::array::from_fn(|_| (rng.random::<f64>() * 2.0 - 1.0) * limit)
            }),
            biases: [0.0; OUT_SIZE],
        }
    }
}

impl<const IN_SIZE: usize, const OUT_SIZE: usize> Layer<IN_SIZE, OUT_SIZE> {
    pub fn feed_forward(&self, input: &[f64; IN_SIZE]) -> [f64; OUT_SIZE] {
        let mut output = [0.; OUT_SIZE];

        for (i, n) in output.iter_mut().enumerate() {
            let scalar_product: f64 = input
                .iter()
                .zip(self.weights[i].iter())
                .map(|(x, y)| x * y)
                .sum();

            *n = sigmoid(scalar_product + self.biases[i]);
        }

        output
    }

    pub fn update_layer(
        &mut self,
        delta: &[f64; OUT_SIZE],
        input_activation: &[f64; IN_SIZE],
        learning_rate: f64,
    ) {
        for j in 0..OUT_SIZE {
            self.biases[j] -= learning_rate * delta[j];

            for k in 0..IN_SIZE {
                self.weights[j][k] -= learning_rate * delta[j] * input_activation[k];
            }
        }
    }

    pub fn previous_delta(
        &self,
        current_delta: &[f64; OUT_SIZE],
        previous_activation: &[f64; IN_SIZE],
    ) -> [f64; IN_SIZE] {
        std::array::from_fn(|prev_neuron_idx| {
            let weighted_sum: f64 = self
                .weights
                .iter()
                .zip(current_delta.iter())
                .map(|(w, d)| w[prev_neuron_idx] * d)
                .sum();

            weighted_sum * sigmoid_derivate(previous_activation[prev_neuron_idx])
        })
    }
}
