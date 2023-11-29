//! src/layer.rs

use crate::{neuron::Neuron, value::Value};

#[derive(Clone, Debug)]
pub struct Layer(Vec<Neuron>);

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Layer {
        let neurons = (0..nout).map(|_| Neuron::new(nin)).collect();

        Layer(neurons)
    }

    pub fn call(&self, inputs: &Vec<Value>) -> Vec<Value> {
        self.neurons().iter().map(|n| n.call(inputs)).collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons()
            .iter()
            .map(|n| n.parameters())
            .collect::<Vec<Vec<Value>>>()
            .concat()
    }

    pub fn neurons(&self) -> &Vec<Neuron> {
        &self.0
    }

    pub fn set_neurons(&mut self, neurons: Vec<Neuron>) {
        self.0 = neurons;
    }
}

#[cfg(test)]
mod tests {
    use crate::value::Value;

    use super::Layer;

    #[test]
    fn test_init_layer() {
        let x1 = Value::new(2.0);
        let x2 = Value::new(0.0);
        let layer = Layer::new(2, 3);

        let out = layer.call(&vec![x1, x2]);

        println!("value outs: ");
        out.iter().for_each(|v| print!("{}, ", v));
        print!(")");

        assert_eq!(out.len(), 3);
    }
}
