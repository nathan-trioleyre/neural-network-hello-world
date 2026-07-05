use crate::math::sigmoid;
use rand::RngExt;

pub struct Layer<const IN_SIZE: usize, const OUT_SIZE: usize> {
    weights: [[f64; IN_SIZE]; OUT_SIZE],
    biases: [f64; OUT_SIZE],
}

impl<const IN_SIZE: usize, const OUT_SIZE: usize> Default for Layer<IN_SIZE, OUT_SIZE> {
    fn default() -> Self {
        let mut rng = rand::rng();
        let limit = (6.0 / (IN_SIZE + OUT_SIZE) as f64).sqrt();

        Self {
            weights: std::array::from_fn(|_| std::array::from_fn(|_| (rng.random::<f64>() * 2.0 - 1.0) * limit)),
            biases: [0.0; OUT_SIZE],
        }
    }
}

impl<const IN_SIZE: usize, const OUT_SIZE: usize> Layer<IN_SIZE, OUT_SIZE> {
	pub fn feed_forward(&self, input: [f64; IN_SIZE]) -> [f64; OUT_SIZE] {
		let mut output = [0.; OUT_SIZE];

		for (i, n) in output.iter_mut().enumerate() {
			let scalar_product: f64 = input.iter()
				.zip(self.weights[i])
				.map(|(x, y)| x * y)
				.sum();

			*n = sigmoid(scalar_product + self.biases[i]);
		}

		output
	}
}
