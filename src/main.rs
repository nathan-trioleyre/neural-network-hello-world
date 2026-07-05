use crate::neural_network::NeuralNetwork;

mod layer;
mod math;
mod neural_network;


fn main() {
    let _network = NeuralNetwork::default();
    println!("Initialied a random neural network!");
}
