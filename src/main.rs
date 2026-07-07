use mnist::{Mnist, MnistBuilder};

use crate::neural_network::{NeuralNetwork, ImagesSet};

mod layer;
mod math;
mod neural_network;

fn to_gray_image(image: &[u8]) -> [f64; 784] {
	std::array::from_fn(|i| (image[i] as f64) / 255.0)
}

fn build_images(images: Vec<u8>) -> Vec<[f64; 784]> {
	images
		.chunks_exact(784)
		.take(IMAGES_COUNT)
		.map(to_gray_image)
		.collect()
}

const IMAGES_COUNT: usize = 35_000;

fn main() {
    let mut network = NeuralNetwork::default();
    
	let Mnist {
		tst_img,
		tst_lbl,
		trn_img,
		trn_lbl,
		..
	} = MnistBuilder::new()
		.label_format_digit()
		.test_set_length(IMAGES_COUNT as u32)
		.training_set_length(IMAGES_COUNT as u32)
		.finalize();

	let test_images: Vec<[f64; 784]> = build_images(tst_img);

	let test_set = ImagesSet::new(test_images, tst_lbl);

	println!("Cost before training: {}", network.cost(&test_set));

	let train_images = build_images(trn_img);

	let train_set = ImagesSet::new(train_images, trn_lbl);

	network.train(&train_set);

	println!("Cost after training: {}", network.cost(&test_set));
}
