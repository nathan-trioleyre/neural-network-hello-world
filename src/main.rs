mod app;
mod layer;
mod math;
mod mnist_dataset;
mod neural_network;

use color_eyre::eyre::Result;

use crate::app::App;

fn main() -> Result<()> {
    color_eyre::install()?;
    ratatui::run(|terminal| App::default().run(terminal))
}
