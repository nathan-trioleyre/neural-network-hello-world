use mnist::{Mnist, MnistBuilder};
use rand::RngExt;

pub struct ImageSet {
    pub images: Vec<[f64; 784]>,
    pub labels: Vec<u8>,
}

const DATASET_LIMIT: u32 = 10_000;

impl ImageSet {
    pub fn select_random_digit(&self, digit: u8) -> Option<usize> {
        let indices: Vec<usize> = self.labels.iter()
            .enumerate()
            .filter(|&(_, &label)| label == digit)
            .map(|(index, _)| index)
            .collect();
        if indices.is_empty() {
            None
        } else {
            let mut rng = rand::rng();
            Some(indices[rng.random_range(0..indices.len())])
        }
    }

    pub fn build_testing_set() -> Self {
        let Mnist {
            tst_img, tst_lbl, ..
        } = MnistBuilder::new()
            .label_format_digit()
            .test_set_length(DATASET_LIMIT)
            .finalize();

        Self {
            images: ImageSet::convert_images(tst_img),
            labels: tst_lbl,
        }
    }

    pub fn build_training_set() -> Self {
        let Mnist {
            trn_img, trn_lbl, ..
        } = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(DATASET_LIMIT)
            .finalize();

        Self {
            images: ImageSet::convert_images(trn_img),
            labels: trn_lbl,
        }
    }

    fn convert_images(images: Vec<u8>) -> Vec<[f64; 784]> {
        images
            .chunks_exact(784)
            .take(DATASET_LIMIT as usize)
            .map(|image| std::array::from_fn(|i| (image[i] as f64) / 255.))
            .collect()
    }
}
