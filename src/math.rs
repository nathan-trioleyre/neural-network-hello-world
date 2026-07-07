pub fn sigmoid(z: f64) -> f64 {
    1. / (1. + (-z).exp())
}

pub fn sigmoid_derivation(a: f64) -> f64 {
	a * (1. - a)
}
