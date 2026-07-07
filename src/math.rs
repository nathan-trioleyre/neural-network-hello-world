pub fn sigmoid(z: f64) -> f64 {
    1. / (1. + (-z).exp())
}

pub fn sigmoid_derivate(activation: f64) -> f64 {
    activation * (1. - activation)
}
