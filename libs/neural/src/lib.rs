/// This first implementation is incredibly object oriented just to allow me to model it like I visualize it.
/// Once we get this working, we'll adapt it to be matrix multiplication and much more efficient

pub mod mnist;

use std::cmp::max;
use std::collections::HashMap;
use std::rc::Rc;
use crate::mnist::MnistSample;

pub trait Sample {}

// Trait for loading batches of data into a NeuralNetwork for training runs
pub trait DataLoader: Iterator<Item = Vec<Self::SampleType>> {
    type SampleType: Sample;
    fn reset(&mut self);
}

pub struct NeuralNet {
    /// TODO this isn't a "layer" its just your data, but the weights are definetly real
    pub input_layer: Layer,
    pub hidden_layers: Vec<Layer>,
    pub output_layer: Layer,
    pub weights: HashMap<(Rc<Neuron>, Rc<Neuron>),f64>,
}

/// TODO better typesafety on relu with boolean output
fn relu(input:usize) -> usize {
    max(0, input)
}

/// TODO make generic to Samples
/// TODO make the arrays and everything type safe fixed length
impl NeuralNet {
    pub fn forward_pass(&self, mnist_sample: MnistSample) -> usize {
        for (index, layer) in self.hidden_layers.iter().enumerate() {
            for destination_neuron in layer.neurons {
                if(index == 0) {
                    let mut neuron_input = 0;
                    for i in self.input_layer.neurons {
                        neuron_input += mnist_sample.pixels().get(i) * self.weights.get(&(i.clone(), destination_neuron.clone()))

                    }
                } else {
                    0
                }
            }
        }
    }


    /// General algorithm needs to be:
    ///  Take a batch size b and a data loader
    ///  Until we run out of batches
    ///     Load b samples s from the data loader
    ///     Run gradient descent on s
    ///     Update some OpenTelemetry metrics so we know what's going on, for example loss
    pub fn train<D: DataLoader>(&mut self, mut loader: D) -> () {
        for batch in loader {

        }
    }

    pub fn layers(&self) -> &[Layer] {
        &self.hidden_layers
    }
}

impl NeuralNet {
    pub fn new(input_layer_size: usize, num_hidden_layers: usize, hidden_layer_size: usize, output_layer_size: usize) -> Self {
        let hidden_layers: Vec<Layer> = (0..num_hidden_layers).map(|_| Layer::new(hidden_layer_size)).collect();
        let mut weights: HashMap<(Rc<Neuron>, Rc<Neuron>), Weight> = HashMap::new();

        for index in 0..hidden_layers.len() - 1 {
            let current_layer = &hidden_layers[index];
            let next_layer = &hidden_layers[index + 1];

            for neuron in &current_layer.neurons {
                for dest_neuron in &next_layer.neurons {
                    weights.insert((Rc::clone(neuron), Rc::clone(dest_neuron)), 0.0);
                }
            }
        }

        let input_layer: Layer = Layer::new(input_layer_size);

        for neuron in &input_layer.neurons {
            for dest_neuron in &hidden_layers.get(0).unwrap().neurons {
                weights.push(Weight {
                    source: Rc::clone(neuron),
                    destination: Rc::clone(dest_neuron),
                    weight: 0.0,
                })
            }
        }

        let output_layer: Layer = Layer::new(output_layer_size);

        for neuron in &output_layer.neurons {
            for dest_neuron in &hidden_layers.get(hidden_layers.len()-1).unwrap().neurons {
                weights.push(Weight {
                    source: Rc::clone(neuron),
                    destination: Rc::clone(dest_neuron),
                    weight: 0.0,
                })
            }
        }


        NeuralNet { input_layer, hidden_layers, output_layer, weights }
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
    pub bias: usize,
}

impl Neuron {
    pub fn new(bias: usize) -> Self {
        Neuron { bias }
    }

    pub fn bias(&self) -> usize {
        self.bias
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use crate::{Neuron, NeuralNet};
    use crate::mnist::MnistLoader;

    #[test]
    fn test_nn_construction() {
        NeuralNet::new(784, 3, 28, 10, );
    }

    #[test]
    fn test_nn_training() {
        let mut neural_net = NeuralNet::new(784, 3, 28, 10);

        let loader = MnistLoader::from_file(
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../perceptron/digit-recognizer/train.csv"),
            32
        ).unwrap();

        assert!(loader.len() > 0);

        neural_net.train(loader);
    }

}