mod app;
mod mnist_dataset;
mod neural;
mod trainer;
mod ui;

use color_eyre::eyre::Result;

use crate::app::App;

fn main() -> Result<()> {
    color_eyre::install()?;
    ratatui::run(|terminal| App::default().run(terminal))
}
