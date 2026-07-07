use color_eyre::eyre::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use rand::RngExt;
use ratatui::{
    DefaultTerminal, Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Stylize},
    symbols::Marker,
    text::Line,
    widgets::{
        Block, Gauge, Paragraph, Sparkline, Tabs,
        canvas::{Canvas, Map, MapResolution, Points},
    },
};

use crate::{mnist_dataset::ImageSet, neural_network::NeuralNetwork};

pub struct App {
    network: NeuralNetwork,
    testing_set: ImageSet,
    training_set: ImageSet,
    selected_digit: u8,
    prediction: Option<u8>,
    active_tab: usize,
    should_exit: bool,
}

impl Default for App {
    fn default() -> Self {
        Self {
            network: Default::default(),
            testing_set: ImageSet::build_testing_set(),
            training_set: ImageSet::build_training_set(),
            selected_digit: Default::default(),
            prediction: None,
            active_tab: Default::default(),
            should_exit: Default::default(),
        }
    }
}

impl App {
    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        while !self.should_exit {
            terminal.draw(|frame| self.draw(frame))?;
            self.handle_events()?;
        }

        Ok(())
    }

    fn handle_events(&mut self) -> Result<()> {
        match event::read()? {
            Event::Key(key_event) if key_event.kind == KeyEventKind::Press => {
                self.handle_key_event(key_event)
            }
            _ => {}
        };

        Ok(())
    }

    fn handle_key_event(&mut self, key_event: KeyEvent) {
        match key_event.code {
            KeyCode::Char('q') => self.should_exit = true,
            KeyCode::Char('r') if self.active_tab == 0 => {
                let mut rng = rand::rng();
                self.selected_digit = rng.random_range(0..10);
            }
            KeyCode::Char('1') => self.active_tab = 0,
            KeyCode::Enter if self.active_tab == 0 => {
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
            KeyCode::Up if self.active_tab == 0 => {
                self.selected_digit = (self.selected_digit + 1) % 10;
            }
            KeyCode::Down if self.active_tab == 0 => {
                self.selected_digit = (self.selected_digit + 9) % 10;
            }
            _ => {}
        }
    }

    fn exit(&mut self) {
        self.should_exit = true;
    }

    fn draw(&self, frame: &mut Frame) {
        let chunks =
            Layout::vertical([Constraint::Length(3), Constraint::Min(0)]).split(frame.area());

        self.draw_header(frame, chunks[0]);

        match self.active_tab {
            0 => self.draw_predict_tab(frame, chunks[1]),
            _ => unreachable!(),
        }
    }

    fn draw_header(&self, frame: &mut Frame, area: Rect) {
        let tabs_title = Line::from(vec![" Hello ".blue(), "Neural ".white(), "Network! ".red()]);

        let tabs = Tabs::new(vec!["[1] Predict", "[2] Train", "[3] Cost"])
            .select(self.active_tab)
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
}
