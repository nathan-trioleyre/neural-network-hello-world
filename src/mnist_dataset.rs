use mnist::{Mnist, MnistBuilder};

pub struct ImageSet {
    pub images: Vec<[f64; 784]>,
    pub labels: Vec<u8>,
}

const DATASET_LIMIT: u32 = 100;

impl ImageSet {
    pub fn select_digit(&self, digit: u8) -> Option<usize> {
        self.labels.iter().position(|&label| label == digit)
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
