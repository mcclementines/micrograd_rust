//! src/mlp.rs

use crate::{layer::Layer, value::Value};

#[derive(Clone, Debug)]
pub struct Mlp(Vec<Layer>);

impl Mlp {
    pub fn new(mut nin: usize, nouts: Vec<usize>) -> Mlp {
        let layers = nouts
            .iter()
            .map(|l| {
                let layer = Layer::new(nin, *l);
                nin = *l;
                layer
            })
            .collect();

        Mlp(layers)
    }

    pub fn call(&self, inputs: &Vec<Value>) -> Vec<Value> {
        let mut output: Vec<Value> = inputs.clone();

        self.layers().iter().for_each(|l| {
            output = l.call(&output);
        });

        output
    }

    pub fn callf(&self, inputs: &Vec<f32>) -> Vec<Value> {
        let inputs = inputs.iter().map(|v| Value::new(v.clone())).collect();
        
        self.call(&inputs)
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers()
            .iter()
            .map(|l| l.parameters())
            .collect::<Vec<Vec<Value>>>()
            .concat()
    }

    pub fn layers(&self) -> &Vec<Layer> {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::Mlp;

    #[test]
    fn test_mlp_init_and_call() {
        let layers = vec![3, 3, 1];
        let mlp = Mlp::new(3, layers);

        let inputs = vec![2.0, 3.0, 2.0];

        let out = mlp.callf(&inputs);

        assert_eq!(out.len(), 1);

        println!("mlp out: {:?}", out);
        println!("mlp params: {:?}", mlp.parameters());
    }
}
