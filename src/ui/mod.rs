mod predict;
mod train;

use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::Stylize,
    text::Line,
    widgets::{Block, Tabs},
};

use crate::app::{App, AppTab};

pub fn draw(app: &App, frame: &mut Frame) {
    let chunks = Layout::vertical([Constraint::Length(3), Constraint::Min(0)]).split(frame.area());

    draw_header(app, frame, chunks[0]);

    match app.active_tab {
        AppTab::Prediction => predict::draw_predict_tab(app, frame, chunks[1]),
        AppTab::Training => train::draw_train_tab(app, frame, chunks[1]),
    }
}

fn draw_header(app: &App, frame: &mut Frame, area: Rect) {
    let tabs_title = Line::from(vec![" Hello ".blue(), "Neural ".white(), "Network! ".red()]);

    let tabs = Tabs::new(vec!["[1] Predict", "[2] Train"])
        .select(app.active_tab as usize)
        .block(Block::bordered().title(tabs_title));

    frame.render_widget(tabs, area);
}
