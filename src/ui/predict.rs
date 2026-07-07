use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::{Color, Stylize},
    symbols::Marker,
    text::Line,
    widgets::{
        Block, Paragraph,
        canvas::{Canvas, Points},
    },
};

use crate::app::App;

pub fn draw_predict_tab(app: &App, frame: &mut Frame, area: Rect) {
    let main_chunks =
        Layout::horizontal([Constraint::Percentage(30), Constraint::Percentage(70)]).split(area);

    let canvas = Canvas::default()
        .x_bounds([0., 27.])
        .y_bounds([0., 27.])
        .marker(Marker::HalfBlock)
        .paint(|ctx| {
            let image = app.testing_set.images[app.selected_image_index];

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

    let right_chunks = Layout::vertical([
        Constraint::Length(3), // Info label / key bindings
        Constraint::Min(0),    // Prediction result
    ])
    .split(main_chunks[1]);

    let keys_block = Paragraph::new(
        Line::from(vec![
            " [Enter]".blue().bold(),
            " Predict | ".into(),
            " [r]".blue().bold(),
            " Random Digit | ".into(),
            " [Up]/[Down]".blue().bold(),
            " Choose Digit ".into(),
        ])
        .centered(),
    )
    .block(Block::bordered().title(" Controls "));

    frame.render_widget(keys_block, right_chunks[0]);

    let mut prediction_lines = vec![
        Line::default(),
        Line::from(vec![
            "Selected Digit: ".into(),
            app.selected_digit.to_string().bold(),
        ])
        .centered(),
        Line::default(),
    ];

    if let Some(prediction) = app.prediction {
        let is_correct = prediction == app.selected_digit;
        let color = if is_correct { Color::Green } else { Color::Red };

        prediction_lines.extend(vec![
            Line::from(vec![
                "Prediction: ".into(),
                prediction.to_string().bold().fg(color),
            ])
            .centered(),
            Line::default(),
            Line::from(vec![if is_correct {
                "CORRECT!".green().bold()
            } else {
                "WRONG!".red().bold()
            }])
            .centered(),
        ]);
    } else {
        prediction_lines.extend(vec![
            Line::from("Prediction: N/A".bold()).centered(),
            Line::default(),
            Line::default(),
        ]);
    }

    let prediction_block =
        Paragraph::new(prediction_lines).block(Block::bordered().title(" Prediction "));

    frame.render_widget(prediction_block, right_chunks[1]);
}
