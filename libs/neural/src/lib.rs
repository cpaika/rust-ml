/// This first implementation is incredibly object oriented just to allow me to model it like I visualize it.
/// Once we get this working, we'll adapt it to be matrix multiplication and much more efficient

use std::rc::Rc;

pub struct NeuralNet {
    layers: Vec<Layer>,
    weights: Vec<Weight>,
}

impl NeuralNet {
    pub fn new(num_layers: usize, neurons: usize) -> Self {
        let layers: Vec<Layer> = (0..num_layers).map(|_| Layer::new(neurons)).collect();
        let mut weights: Vec<Weight> = vec![];

        for index in 0..layers.len() - 1 {
            let current_layer = &layers[index];
            let next_layer = &layers[index + 1];

            for neuron in &current_layer.neurons {
                for dest_neuron in &next_layer.neurons {
                    weights.push(Weight {
                        source: Rc::clone(neuron),
                        destination: Rc::clone(dest_neuron),
                        weight: 0.0,
                    })
                }
            }
        }

        NeuralNet { layers, weights }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Layer {
    pub neurons: Vec<Rc<Neuron>>,
}

impl Layer {
    pub fn new(neurons: usize) -> Self {
        Layer {
            neurons: (0..neurons).map(|_| Rc::new(Neuron::new(0))).collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Neuron {
    bias: usize,
}

impl Neuron {
    pub fn new(bias: usize) -> Self {
        Neuron { bias }
    }
}

pub struct Weight {
    source: Rc<Neuron>,
    destination: Rc<Neuron>,
    weight: f64,
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use crate::{Neuron, Weight, NeuralNet};

    #[test]
    fn test_nn_construction() {
        let neural_net = NeuralNet::new(3, 28);
    }

}