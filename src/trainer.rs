use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    thread,
};

use crate::{mnist_dataset::ImageSet, neural::network::NeuralNetwork};

pub struct TrainingUpdate {
    pub current_epoch: usize,
    pub loss: f64,
    pub network: NeuralNetwork,
}

pub struct Trainer {
    keep_training: Arc<AtomicBool>,
    receiver: Option<Receiver<TrainingUpdate>>,
}

impl Default for Trainer {
    fn default() -> Self {
        Self {
            keep_training: Arc::new(AtomicBool::new(false)),
            receiver: None,
        }
    }
}

impl Trainer {
    pub fn start(
        &mut self,
        network: NeuralNetwork,
        training_set: Arc<ImageSet>,
        testing_set: Arc<ImageSet>,
        start_epoch: usize,
        total_epochs: usize,
    ) {
        self.keep_training.store(true, Ordering::SeqCst);
        let (sender, receiver) = mpsc::channel();
        self.receiver = Some(receiver);

        let keep_training_clone = Arc::clone(&self.keep_training);
        let mut training_network = network;

        thread::spawn(move || {
            for epoch in start_epoch..total_epochs {
                if !keep_training_clone.load(Ordering::SeqCst) {
                    break;
                }

                training_network.train(&training_set);
                let loss = training_network.loss(&testing_set);

                if sender
                    .send(TrainingUpdate {
                        current_epoch: epoch + 1,
                        loss,
                        network: training_network.clone(),
                    })
                    .is_err()
                {
                    break;
                }
            }
        });
    }

    pub fn pause(&mut self) {
        self.keep_training.store(false, Ordering::SeqCst);
    }

    pub fn try_recv(&self) -> Option<TrainingUpdate> {
        let mut latest_update = None;
        if let Some(ref receiver) = self.receiver {
            while let Ok(update) = receiver.try_recv() {
                latest_update = Some(update);
            }
        }
        latest_update
    }
}
