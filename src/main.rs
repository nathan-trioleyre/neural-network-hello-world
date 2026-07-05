use mnist::{Mnist, MnistBuilder};

use crate::neural_network::NeuralNetwork;

mod layer;
mod math;
mod neural_network;

fn argmax(output_layer: [f64; 10]) -> u8 {
    output_layer
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index as u8)
        .unwrap_or(0)
}

fn main() {
    let network = NeuralNetwork::default();
    
	let Mnist {
		tst_img,
		tst_lbl,
		..
	} = MnistBuilder::new()
		.label_format_digit()
		.validation_set_length(1)
		.test_set_length(1)
		.finalize();

	let mut image = [0.; 784];

	for i in 0..784 {
		image[i] = (tst_img[i] as f64) / 256.;
	}

	println!("{:?}", tst_lbl);

	println!("Predicted number: {}", argmax(network.predict(image)));
	println!("Expected number: {}", tst_lbl[0]);
}
