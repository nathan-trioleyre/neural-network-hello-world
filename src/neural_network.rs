use crate::{layer::Layer, math::sigmoid_derivation};

pub struct NeuralNetwork {
    hidden_layer: Layer<784, 16>,
    second_hidden_layer: Layer<16, 16>,
    output_layer: Layer<16, 10>,
	expected_outputs: [[f64; 10]; 10],
}

fn build_expected_output(expected: usize) -> [f64; 10] {
	std::array::from_fn(|i| if i == expected { 1.} else { 0. })
}

impl Default for NeuralNetwork {
	fn default() -> Self {
		let expected_outputs: [[f64; 10]; 10] = std::array::from_fn(|i| build_expected_output(i));

		Self {
			hidden_layer: Default::default(),
			second_hidden_layer: Default::default(),
			output_layer: Default::default(),
			expected_outputs
		}
	}
}

struct ForwardResult {
	a1: [f64; 16],
	a2: [f64; 16],
	a3: [f64; 10]
}

pub struct ImagesSet {
	images: Vec<[f64; 784]>,
	expected: Vec<u8>
}

impl ImagesSet {
	pub fn new(images: Vec<[f64; 784]>, expected: Vec<u8>) -> Self {
		Self {
			images,
			expected
		}
	}
}

impl NeuralNetwork {
	fn feed_forward(&self, image: &[f64; 784]) -> ForwardResult {
		let a1 = self.hidden_layer.feed_forward(image);
        let a2 = self.second_hidden_layer.feed_forward(&a1);

		ForwardResult {
			a1,
			a2,
			a3: self.output_layer.feed_forward(&a2)
		}
	}

	fn get_expected_output(&self, expected: u8) -> &[f64; 10] {
		&self.expected_outputs[expected as usize]
	}

    pub fn predict(&self, image: &[f64; 784]) -> [f64; 10] {
        self.feed_forward(image).a3
    }

	pub fn cost(&self, test_set: &ImagesSet) -> f64 {
		let total_cost: f64 = test_set.images.iter()
			.zip(test_set.expected.iter())
			.map(|(image, expected)| self.single_cost(image, self.get_expected_output(*expected)))
			.sum();

		total_cost / test_set.images.len() as f64
	}

	fn single_cost(&self, image: &[f64; 784], expected: &[f64; 10]) -> f64 {
		self.predict(&image).iter()
			.zip(expected.iter())
			.map(|(predicted, expected)| (predicted - expected).powi(2))
			.sum()
	}

	pub fn train(&mut self, test_set: &ImagesSet) {
		let eta = 0.01;

		for (image, expected) in test_set.images.iter().zip(&test_set.expected) {
			let feed = self.feed_forward(image);

			let d3: [f64; 10] = std::array::from_fn(|j| {
				let a = feed.a3[j];
				let e = self.get_expected_output(*expected)[j];
				(a - e) * sigmoid_derivation(a)
			});

			let d2 = self.output_layer.previous_delta(&d3, &feed.a2);
			let d1 = self.second_hidden_layer.previous_delta(&d2, &feed.a1);

			self.output_layer.update_layer(&d3, &feed.a2, eta);
			self.second_hidden_layer.update_layer(&d2, &feed.a1, eta);
			self.hidden_layer.update_layer(&d1, &image, eta);
		}
	}
}
