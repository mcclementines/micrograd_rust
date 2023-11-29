//! src/neuron.rs

use std::{cell::RefCell, rc::Rc};

use rand::{distributions::Uniform, prelude::Distribution};

use crate::value::Value;

#[derive(Debug)]
struct InnerNeuron {
    weights: Vec<Value>,
    bias: Value,
}

#[derive(Clone, Debug)]
pub struct Neuron(Rc<RefCell<InnerNeuron>>);

impl Neuron {
    pub fn new(nin: usize) -> Neuron {
        let uniform = Uniform::new_inclusive(-1.0, 1.0);
        let mut rng = rand::thread_rng();

        let weights: Vec<Value> = (0..nin)
            .map(|_| Value::new(uniform.sample(&mut rng)))
            .collect();
        let bias = Value::new(uniform.sample(&mut rng));

        let neuron = InnerNeuron { weights, bias };

        Neuron(Rc::new(RefCell::new(neuron)))
    }

    pub fn call(&self, inputs: &Vec<Value>) -> Value {
        assert_eq!(
            self.num_weights(),
            inputs.len(),
            "num of inputs ({}) do not equal num of weights ({})",
            inputs.len(),
            self.num_weights()
        );

        let zipped = inputs.iter().zip(self.weights());

        (zipped.map(|(x1, w1)| x1 * &w1).sum::<Value>() + self.bias()).tanh()
    }

    pub fn callf(&self, inputs: &Vec<f32>) -> Value {
        let inputs: Vec<Value> = inputs.iter().map(|x| Value::new(*x)).collect();

        self.call(&inputs)
    }

    pub fn parameters(&self) -> Vec<Value> {
        [self.weights(), vec![self.bias()]].concat()
    }

    pub fn weights(&self) -> Vec<Value> {
        self.0.borrow().weights.clone()
    }

    pub fn num_weights(&self) -> usize {
        self.0.borrow().weights.len()
    }

    pub fn set_weights(&self, weights: Vec<Value>) {
        self.0.borrow_mut().weights = weights;
    }

    pub fn bias(&self) -> Value {
        self.0.borrow().bias.clone()
    }

    pub fn set_bias(&self, bias: Value) {
        self.0.borrow_mut().bias = bias;
    }
}

#[cfg(test)]
mod tests {
    use crate::value::Value;

    use super::Neuron;

    #[test]
    fn test_neuron_call() {
        let x1 = Value::new(2.0);
        let x2 = Value::new(0.0);

        let w1 = Value::new(-3.0);
        let w2 = Value::new(1.0);

        let b = Value::new(6.8813735870195432);

        let neuron = Neuron::new(2);
        neuron.set_weights(vec![w1, w2]);
        neuron.set_bias(b);

        let result = neuron.call(&vec![x1.clone(), x2.clone()]);
        result.backward();

        assert_eq!(result.data(), 0.7071067);
        assert_eq!(neuron.weights().get(0).unwrap().grad(), 1.0000002);
        assert_eq!(neuron.weights().get(1).unwrap().grad(), 0.0);
        assert_eq!(x1.grad(), -1.5000004);
        assert_eq!(x2.grad(), 0.5000001);
    }
}
