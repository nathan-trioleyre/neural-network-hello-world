use std::{
    fmt::Display,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    thread,
    time::Duration,
};

use color_eyre::eyre::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, poll};
use rand::RngExt;
use ratatui::{
    DefaultTerminal, Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Stylize},
    symbols::Marker,
    text::Line,
    widgets::{
        Block, Gauge, Paragraph, Tabs,
        canvas::{Canvas, Points},
    },
};

use crate::{mnist_dataset::ImageSet, neural_network::NeuralNetwork};

#[derive(PartialEq, Clone, Copy)]
enum AppTab {
    Prediction = 0,
    Training = 1,
}

#[derive(PartialEq)]
enum TrainingState {
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

pub struct TrainingUpdate {
    pub current_epoch: usize,
    pub cost: f64,
    pub network: NeuralNetwork,
}

pub struct App {
    network: NeuralNetwork,
    testing_set: Arc<ImageSet>,
    training_set: Arc<ImageSet>,
    active_tab: AppTab,
    should_exit: bool,

    // Predict tab
    selected_digit: u8,
    prediction: Option<u8>,

    // Train tab
    training_state: TrainingState,
    current_epoch: usize,
    total_epochs: usize,
    cost: Option<f64>,
    receiver: Option<Receiver<TrainingUpdate>>,
    is_training: Arc<AtomicBool>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            network: Default::default(),
            testing_set: Arc::new(ImageSet::build_testing_set()),
            training_set: Arc::new(ImageSet::build_training_set()),
            active_tab: AppTab::Prediction,
            should_exit: Default::default(),
            selected_digit: Default::default(),
            prediction: None,
            training_state: TrainingState::NotStarted,
            current_epoch: 0,
            total_epochs: 10,
            cost: None,
            receiver: None,
            is_training: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl App {
    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        while !self.should_exit {
            terminal.draw(|frame| self.draw(frame))?;
            self.handle_events()?;
            self.update();
        }

        Ok(())
    }

    fn update(&mut self) {
        if self.training_state == TrainingState::Running {
            if let Some(receiver) = &self.receiver {
                while let Ok(training_update) = receiver.try_recv() {
                    self.current_epoch = training_update.current_epoch;
                    self.cost = Some(training_update.cost);
                    self.network = training_update.network;
                }

                if self.current_epoch >= self.total_epochs {
                    self.training_state = TrainingState::Finished;
                    self.is_training.store(false, Ordering::SeqCst);
                }
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
            KeyCode::Char('q') => self.should_exit = true,
            KeyCode::Char('r') => {
                if self.active_tab == AppTab::Prediction {
                    let mut rng = rand::rng();
                    self.selected_digit = rng.random_range(0..10);
                }

                if self.active_tab == AppTab::Training {
                    self.network = Default::default();
                }
            }
            KeyCode::Char('1') => self.active_tab = AppTab::Prediction,
            KeyCode::Char('2') => self.active_tab = AppTab::Training,
            KeyCode::Enter => {
                if self.active_tab == AppTab::Prediction {
                    if let Some(index) = self.testing_set.select_digit(self.selected_digit) {
                        let image = self.testing_set.images[index];
                        let output_activations = self.network.predict(&image);

                        self.prediction = output_activations
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.total_cmp(b))
                            .map(|(index, _)| index as u8);
                    }
                }

                if self.active_tab == AppTab::Training {
                    match self.training_state {
                        TrainingState::NotStarted
                        | TrainingState::Paused
                        | TrainingState::Finished => {
                            if self.training_state == TrainingState::Finished {
                                self.current_epoch = 0;
                            }

                            self.training_state = TrainingState::Running;

                            self.is_training.store(true, Ordering::SeqCst);

                            let (sender, receiver) = mpsc::channel::<TrainingUpdate>();

                            self.receiver = Some(receiver);

                            let training_set_clone = Arc::clone(&self.training_set);
                            let mut training_network = self.network.clone();
                            let current_epoch_clone = self.current_epoch;
                            let total_epochs_clone = self.total_epochs;
                            let is_training_clone = Arc::clone(&self.is_training);

                            thread::spawn(move || {
                                for i in current_epoch_clone..total_epochs_clone {
                                    if is_training_clone.load(Ordering::SeqCst) == false {
                                        break;
                                    }

                                    training_network.train(&training_set_clone);
                                    let cost = training_network.cost(&training_set_clone);

                                    if sender
                                        .send(TrainingUpdate {
                                            current_epoch: i + 1,
                                            cost,
                                            network: training_network.clone(),
                                        })
                                        .is_err()
                                    {
                                        break;
                                    }
                                }
                            });
                        }
                        TrainingState::Running => {
                            self.training_state = TrainingState::Paused;
                            self.is_training.store(false, Ordering::SeqCst);
                        }
                    };
                }
            }
            KeyCode::Up if self.active_tab == AppTab::Prediction => {
                self.selected_digit = (self.selected_digit + 1) % 10;
            }
            KeyCode::Down if self.active_tab == AppTab::Prediction => {
                self.selected_digit = (self.selected_digit + 9) % 10;
            }
            _ => {}
        }
    }

    fn draw(&self, frame: &mut Frame) {
        let chunks =
            Layout::vertical([Constraint::Length(3), Constraint::Min(0)]).split(frame.area());

        self.draw_header(frame, chunks[0]);

        match self.active_tab {
            AppTab::Prediction => self.draw_predict_tab(frame, chunks[1]),
            AppTab::Training => self.draw_train_tab(frame, chunks[1]),
        }
    }

    fn draw_header(&self, frame: &mut Frame, area: Rect) {
        let tabs_title = Line::from(vec![" Hello ".blue(), "Neural ".white(), "Network! ".red()]);

        let tabs = Tabs::new(vec!["[1] Predict", "[2] Train"])
            .select(self.active_tab as usize)
            .block(Block::bordered().title(tabs_title));

        frame.render_widget(tabs, area);
    }

    fn draw_predict_tab(&self, frame: &mut Frame, area: Rect) {
        let main_chunks =
            Layout::horizontal([Constraint::Percentage(30), Constraint::Percentage(70)])
                .split(area);

        let canvas = Canvas::default()
            .x_bounds([0., 27.])
            .y_bounds([0., 27.])
            .marker(Marker::HalfBlock)
            .paint(|ctx| {
                if let Some(index) = self.testing_set.select_digit(self.selected_digit) {
                    let image = self.testing_set.images[index];

                    for i in 0..784 {
                        let gray_color = (image[i] * 255.) as u8;

                        ctx.draw(&Points {
                            coords: &[(i as f64 % 28., 27. - (i / 28) as f64)],
                            color: Color::Rgb(gray_color, gray_color, gray_color),
                        });
                    }
                }
            });

        frame.render_widget(
            canvas.block(Block::bordered().title(" Image ".bold())),
            main_chunks[0],
        );

        let bottom_title = Line::from(vec![
            " <Up/Down>".blue(),
            " Select digit | ".into(),
            "[Enter]".blue(),
            " Predict | ".into(),
            "[r]".blue(),
            " Random ".into(),
        ]);

        let prediction_text = format!(
            " Prediction: {} ",
            if let Some(p) = self.prediction {
                p.underlined()
            } else {
                "".into()
            }
        );

        frame.render_widget(
            Paragraph::new(prediction_text).block(
                Block::bordered()
                    .title(format!(" Label: {} ", self.selected_digit).bold())
                    .title_bottom(bottom_title.centered()),
            ),
            main_chunks[1],
        );
    }

    fn draw_train_tab(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::vertical([Constraint::Length(3), Constraint::Min(0)]).split(area);

        let progress_ratio = if self.total_epochs > 0 {
            self.current_epoch as f64 / self.total_epochs as f64
        } else {
            0.0
        };

        let gauge = Gauge::default()
            .block(Block::bordered().title(" Training Progress "))
            .gauge_style(Color::Blue)
            .ratio(progress_ratio.clamp(0.0, 1.0));

        frame.render_widget(gauge, chunks[0]);

        let bottom_chunks =
            Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(chunks[1]);

        let informations = Paragraph::new(vec![
            Line::from(vec![
                "Status: ".into(),
                self.training_state.to_string().bold(),
            ]),
            Line::from(vec![
                "Epoch: ".into(),
                format!("{}/{}", self.current_epoch, self.total_epochs).into(),
            ]),
            Line::from(vec![
                "Learning Rate: ".into(),
                self.network.learning_rate.to_string().into(),
            ]),
        ])
        .block(Block::bordered().title(" Info "));

        frame.render_widget(informations, bottom_chunks[0]);

        let metrics = Paragraph::new(vec![
            Line::from(vec![
                "Current Cost: ".into(),
                if let Some(cost) = self.cost {
                    cost.to_string().into()
                } else {
                    "".into()
                },
            ]),
            Line::from(vec!["Time Elapsed: ".into(), "00:00".into()]),
        ])
        .block(Block::bordered().title(" Metrics "));

        frame.render_widget(metrics, bottom_chunks[1]);
    }
}
