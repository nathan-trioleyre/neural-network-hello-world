use crate::layer::Layer;

#[derive(Default)]
pub struct NeuralNetwork {
    hidden_layer: Layer<784, 16>,
    second_hidden_layer: Layer<16, 16>,
    output_layer: Layer<16, 10>,
}

impl NeuralNetwork {
    pub fn predict(&self, image: [f64; 784]) -> [f64; 10] {
        let hidden_layer = self.hidden_layer.feed_forward(image);
        let second_hidden_layer = self.second_hidden_layer.feed_forward(hidden_layer);

        self.output_layer.feed_forward(second_hidden_layer)
    }
}
