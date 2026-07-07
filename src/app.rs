use std::{
    fmt::Display,
    sync::Arc,
    time::{Duration, Instant},
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
        Block, Gauge, Paragraph, Sparkline, Tabs,
        canvas::{Canvas, Points},
    },
};

use crate::{mnist_dataset::ImageSet, neural_network::NeuralNetwork, trainer::Trainer};

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

pub struct App {
    network: NeuralNetwork,
    testing_set: Arc<ImageSet>,
    training_set: Arc<ImageSet>,
    active_tab: AppTab,
    should_exit: bool,

    // Predict tab
    selected_digit: u8,
    prediction: Option<u8>,
    selected_image_index: usize,

    // Train tab
    training_state: TrainingState,
    current_epoch: usize,
    total_epochs: usize,
    loss: Option<f64>,
    loss_history: Vec<f64>,
    start_time: Option<Instant>,
    elapsed_time: Duration,
    trainer: Trainer,
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
            terminal.draw(|frame| self.draw(frame))?;
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
                let image = self.testing_set.images[self.selected_image_index];

                for i in 0..784 {
                    let gray_color = (image[i] * 255.) as u8;

                    ctx.draw(&Points {
                        coords: &[(i as f64 % 28., 27. - (i / 28) as f64)],
                        color: Color::Rgb(gray_color, gray_color, gray_color),
                    });
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
        // Découpage vertical : barre de progression (3), zone centrale (variable), aide de contrôles (3)
        let chunks = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .split(area);

        let progress_ratio = if self.total_epochs > 0 {
            self.current_epoch as f64 / self.total_epochs as f64
        } else {
            0.0
        };

        let gauge_color = match self.training_state {
            TrainingState::Running => Color::Blue,
            TrainingState::Paused => Color::Yellow,
            TrainingState::Finished => Color::Green,
            TrainingState::NotStarted => Color::Red,
        };

        let gauge = Gauge::default()
            .block(Block::bordered().title(" Training Progress "))
            .gauge_style(gauge_color)
            .ratio(progress_ratio.clamp(0.0, 1.0));

        frame.render_widget(gauge, chunks[0]);

        // Zone centrale découpée en 3 colonnes horizontales
        let bottom_chunks = Layout::horizontal([
            Constraint::Percentage(30),
            Constraint::Percentage(30),
            Constraint::Percentage(40),
        ])
        .split(chunks[1]);

        let informations = Paragraph::new(vec![
            Line::from(vec![
                "Status: ".into(),
                match self.training_state {
                    TrainingState::Running => self.training_state.to_string().blue().bold(),
                    TrainingState::Paused => self.training_state.to_string().yellow().bold(),
                    TrainingState::Finished => self.training_state.to_string().green().bold(),
                    TrainingState::NotStarted => self.training_state.to_string().red().bold(),
                },
            ]),
            Line::from(vec![
                "Epoch: ".into(),
                format!("{}/{}", self.current_epoch, self.total_epochs).into(),
            ]),
            Line::from(vec![
                "Learning Rate: ".into(),
                format!("{:.5}", self.network.learning_rate).into(),
            ]),
        ])
        .block(Block::bordered().title(" Info "));

        frame.render_widget(informations, bottom_chunks[0]);

        // Calcul dynamique du chrono
        let total_duration = self.elapsed_time
            + self
                .start_time
                .map(|start| start.elapsed())
                .unwrap_or(Duration::ZERO);
        let seconds = total_duration.as_secs() % 60;
        let minutes = (total_duration.as_secs() / 60) % 60;
        let millis = (total_duration.as_millis() % 1000) / 100;
        let time_str = format!("{:02}:{:02}.{}", minutes, seconds, millis);

        let metrics = Paragraph::new(vec![
            Line::from(vec![
                "Current Loss: ".into(),
                if let Some(loss) = self.loss {
                    format!("{:.6}", loss).red()
                } else {
                    "N/A".into()
                },
            ]),
            Line::from(vec!["Time Elapsed: ".into(), time_str.into()]),
        ])
        .block(Block::bordered().title(" Metrics "));

        frame.render_widget(metrics, bottom_chunks[1]);

        // Préparation du Sparkline pour le graphique de perte (défilement automatique vers les valeurs récentes)
        let sparkline_width = (bottom_chunks[2].width as usize).saturating_sub(2);
        let start_index = self.loss_history.len().saturating_sub(sparkline_width);
        let sparkline_data: Vec<u64> = self.loss_history[start_index..]
            .iter()
            .map(|&loss| (loss * 1000.0) as u64)
            .collect();

        let sparkline = Sparkline::default()
            .block(Block::bordered().title(" Loss History "))
            .data(&sparkline_data)
            .style(Color::Yellow);

        frame.render_widget(sparkline, bottom_chunks[2]);

        // Zone d'aide et contrôles tout en bas
        let bottom_title = Line::from(vec![
            " [Enter]".blue().bold(),
            " Play/Pause | ".into(),
            "[-] / [+]".blue().bold(),
            " LR | ".into(),
            " [Up] / [Down]".blue().bold(),
            " Epochs (PgUp/PgDn for +/-10) | ".into(),
            "[r]".blue().bold(),
            " Reset ".into(),
        ]);
        let help_block =
            Paragraph::new(bottom_title.centered()).block(Block::bordered().title(" Controls "));
        frame.render_widget(help_block, chunks[2]);
    }
}
