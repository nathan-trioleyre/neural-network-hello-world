use std::{
    fmt::Display,
    sync::Arc,
    time::{Duration, Instant},
};

use color_eyre::eyre::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, poll};
use rand::RngExt;
use ratatui::DefaultTerminal;

use crate::{mnist_dataset::ImageSet, neural::network::NeuralNetwork, trainer::Trainer};

#[derive(PartialEq, Clone, Copy)]
pub enum AppTab {
    Prediction = 0,
    Training = 1,
}

#[derive(PartialEq, Clone, Copy)]
pub enum TrainingState {
    NotStarted,
    Running,
    Paused,
    Finished,
}

impl Display for TrainingState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                TrainingState::NotStarted => "NOT STARTED",
                TrainingState::Running => "RUNNING",
                TrainingState::Paused => "PAUSED",
                TrainingState::Finished => "FINISHED",
            }
        )
    }
}

pub struct App {
    pub network: NeuralNetwork,
    pub testing_set: Arc<ImageSet>,
    pub training_set: Arc<ImageSet>,
    pub active_tab: AppTab,
    pub should_exit: bool,

    // Predict tab
    pub selected_digit: u8,
    pub prediction: Option<u8>,
    pub selected_image_index: usize,

    // Train tab
    pub training_state: TrainingState,
    pub current_epoch: usize,
    pub total_epochs: usize,
    pub loss: Option<f64>,
    pub loss_history: Vec<f64>,
    pub start_time: Option<Instant>,
    pub elapsed_time: Duration,
    pub trainer: Trainer,
}

impl Default for App {
    fn default() -> Self {
        let testing_set = Arc::new(ImageSet::build_testing_set());
        let selected_digit = 0;
        let selected_image_index = testing_set.select_random_digit(selected_digit).unwrap_or(0);

        Self {
            network: Default::default(),
            testing_set,
            training_set: Arc::new(ImageSet::build_training_set()),
            active_tab: AppTab::Prediction,
            should_exit: Default::default(),
            selected_digit,
            prediction: None,
            selected_image_index,
            training_state: TrainingState::NotStarted,
            current_epoch: 0,
            total_epochs: 10,
            loss: None,
            loss_history: Vec::new(),
            start_time: None,
            elapsed_time: Duration::ZERO,
            trainer: Default::default(),
        }
    }
}

impl App {
    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        while !self.should_exit {
            terminal.draw(|frame| crate::ui::draw(self, frame))?;
            self.handle_events()?;
            self.update();
        }

        Ok(())
    }

    fn update(&mut self) {
        if self.training_state == TrainingState::Running {
            if let Some(training_update) = self.trainer.try_recv() {
                self.current_epoch = training_update.current_epoch;
                self.loss = Some(training_update.loss);
                self.loss_history.push(training_update.loss);
                self.network = training_update.network;
            }

            if self.current_epoch >= self.total_epochs {
                self.training_state = TrainingState::Finished;
                if let Some(start) = self.start_time.take() {
                    self.elapsed_time += start.elapsed();
                }
                self.trainer.pause();
            }
        }
    }

    fn handle_events(&mut self) -> Result<()> {
        if poll(Duration::from_millis(10))? {
            match event::read()? {
                Event::Key(key_event) if key_event.kind == KeyEventKind::Press => {
                    self.handle_key_event(key_event)
                }
                _ => {}
            };
        }

        Ok(())
    }

    fn handle_key_event(&mut self, key_event: KeyEvent) {
        match key_event.code {
            KeyCode::Char('q') => {
                self.trainer.pause();
                self.should_exit = true;
            }
            KeyCode::Char('r') => {
                if self.active_tab == AppTab::Prediction {
                    let mut rng = rand::rng();
                    self.selected_digit = rng.random_range(0..10);
                    if let Some(index) = self.testing_set.select_random_digit(self.selected_digit) {
                        self.selected_image_index = index;
                        self.prediction = None;
                    }
                }

                if self.active_tab == AppTab::Training {
                    self.trainer.pause();
                    self.network = Default::default();
                    self.current_epoch = 0;
                    self.loss = None;
                    self.loss_history.clear();
                    self.start_time = None;
                    self.elapsed_time = Duration::ZERO;
                    self.training_state = TrainingState::NotStarted;
                }
            }
            KeyCode::Char('1') => self.active_tab = AppTab::Prediction,
            KeyCode::Char('2') => self.active_tab = AppTab::Training,
            KeyCode::Char('+') | KeyCode::Char('=') if self.active_tab == AppTab::Training => {
                self.network.learning_rate = (self.network.learning_rate * 1.5).min(1.0);
            }
            KeyCode::Char('-') | KeyCode::Char('_') if self.active_tab == AppTab::Training => {
                self.network.learning_rate = (self.network.learning_rate / 1.5).max(0.0001);
            }
            KeyCode::Up => {
                if self.active_tab == AppTab::Prediction {
                    self.selected_digit = (self.selected_digit + 1) % 10;
                    if let Some(index) = self.testing_set.select_random_digit(self.selected_digit) {
                        self.selected_image_index = index;
                        self.prediction = None;
                    }
                }
                if self.active_tab == AppTab::Training
                    && self.training_state != TrainingState::Running
                {
                    self.total_epochs = (self.total_epochs + 1).min(1000);
                }
            }
            KeyCode::Down => {
                if self.active_tab == AppTab::Prediction {
                    self.selected_digit = (self.selected_digit + 9) % 10;
                    if let Some(index) = self.testing_set.select_random_digit(self.selected_digit) {
                        self.selected_image_index = index;
                        self.prediction = None;
                    }
                }
                if self.active_tab == AppTab::Training
                    && self.training_state != TrainingState::Running
                {
                    self.total_epochs = (self.total_epochs - 1).max(1);
                }
            }
            KeyCode::PageUp if self.active_tab == AppTab::Training => {
                if self.training_state != TrainingState::Running {
                    self.total_epochs = (self.total_epochs + 10).min(1000);
                }
            }
            KeyCode::PageDown if self.active_tab == AppTab::Training => {
                if self.training_state != TrainingState::Running {
                    self.total_epochs = (self.total_epochs - 10).max(1);
                }
            }
            KeyCode::Enter => {
                if self.active_tab == AppTab::Prediction {
                    let image = self.testing_set.images[self.selected_image_index];
                    let output_activations = self.network.predict(&image);

                    self.prediction = output_activations
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .map(|(index, _)| index as u8);
                }

                if self.active_tab == AppTab::Training {
                    match self.training_state {
                        TrainingState::NotStarted | TrainingState::Paused => {
                            self.training_state = TrainingState::Running;
                            self.start_time = Some(Instant::now());
                            self.trainer.start(
                                self.network.clone(),
                                Arc::clone(&self.training_set),
                                Arc::clone(&self.testing_set),
                                self.current_epoch,
                                self.total_epochs,
                            );
                        }
                        TrainingState::Running => {
                            self.training_state = TrainingState::Paused;
                            if let Some(start) = self.start_time.take() {
                                self.elapsed_time += start.elapsed();
                            }
                            self.trainer.pause();
                        }
                        TrainingState::Finished => {
                            self.current_epoch = 0;
                            self.elapsed_time = Duration::ZERO;

                            self.training_state = TrainingState::Running;
                            self.start_time = Some(Instant::now());
                            self.trainer.start(
                                self.network.clone(),
                                Arc::clone(&self.training_set),
                                Arc::clone(&self.testing_set),
                                self.current_epoch,
                                self.total_epochs,
                            );
                        }
                    };
                }
            }
            _ => {}
        }
    }
}
