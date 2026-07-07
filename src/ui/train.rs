use std::time::Duration;

use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Stylize},
    text::Line,
    widgets::{Block, Gauge, Paragraph, Sparkline},
};

use crate::app::{App, TrainingState};

pub fn draw_train_tab(app: &App, frame: &mut Frame, area: Rect) {
    let chunks = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(0),
        Constraint::Length(3),
    ])
    .split(area);

    let progress_ratio = if app.total_epochs > 0 {
        app.current_epoch as f64 / app.total_epochs as f64
    } else {
        0.0
    };

    let gauge_color = match app.training_state {
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

    let bottom_chunks = Layout::horizontal([
        Constraint::Percentage(30),
        Constraint::Percentage(30),
        Constraint::Percentage(40),
    ])
    .split(chunks[1]);

    let informations = Paragraph::new(vec![
        Line::from(vec![
            "Status: ".into(),
            match app.training_state {
                TrainingState::Running => app.training_state.to_string().blue().bold(),
                TrainingState::Paused => app.training_state.to_string().yellow().bold(),
                TrainingState::Finished => app.training_state.to_string().green().bold(),
                TrainingState::NotStarted => app.training_state.to_string().red().bold(),
            },
        ]),
        Line::from(vec![
            "Epoch: ".into(),
            format!("{}/{}", app.current_epoch, app.total_epochs).into(),
        ]),
        Line::from(vec![
            "Learning Rate: ".into(),
            format!("{:.5}", app.network.learning_rate).into(),
        ]),
    ])
    .block(Block::bordered().title(" Info "));

    frame.render_widget(informations, bottom_chunks[0]);

    let total_duration = app.elapsed_time
        + app
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
            if let Some(loss) = app.loss {
                format!("{:.6}", loss).red()
            } else {
                "N/A".into()
            },
        ]),
        Line::from(vec!["Time Elapsed: ".into(), time_str.into()]),
    ])
    .block(Block::bordered().title(" Metrics "));

    frame.render_widget(metrics, bottom_chunks[1]);

    let sparkline_width = (bottom_chunks[2].width as usize).saturating_sub(2);
    let start_index = app.loss_history.len().saturating_sub(sparkline_width);
    let sparkline_data: Vec<u64> = app.loss_history[start_index..]
        .iter()
        .map(|&loss| (loss * 1000.0) as u64)
        .collect();

    let sparkline = Sparkline::default()
        .block(Block::bordered().title(" Loss History "))
        .data(&sparkline_data)
        .style(Color::Yellow);

    frame.render_widget(sparkline, bottom_chunks[2]);

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
