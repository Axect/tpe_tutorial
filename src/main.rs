use peroxide::fuga::*;
use rand::SeedableRng;
use tpe::{categorical_range, histogram_estimator, parzen_estimator, range, TpeOptimizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let choices = [1, 2, 3];
    let mut optim0 = TpeOptimizer::new(parzen_estimator(), range(-5.0, 5.0)?);
    let mut optim1 = TpeOptimizer::new(histogram_estimator(), categorical_range(choices.len())?);

    let n_trial = 500;

    let mut best_params = (0.0, 0);
    let mut best_trial = 0usize;
    let mut best_value = std::f64::INFINITY;
    let mut value_history = vec![0f64; n_trial];
    let mut param_history = vec![(0.0, 0.0); n_trial];
    let mut best_value_history = vec![];

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for i in 0..n_trial {
        let x = optim0.ask(&mut rng)?;
        let y = optim1.ask(&mut rng)?;

        let v = objective(x, choices[y as usize]);
        optim0.tell(x, v)?;
        optim1.tell(y, v)?;

        if v < best_value {
            best_params = (x, choices[y as usize]);
            best_trial = i;
            best_value = v;
            best_value_history.push((best_trial as f64, best_value));
        }

        value_history[i] = v;
        param_history[i] = (x, choices[y as usize] as f64);
    }

    println!("Best params: {:?}", best_params);
    println!("Best trial: {}", best_trial);
    println!("Best value: {}", best_value);

    let (x_history, y_history) = param_history.into_iter().unzip();
    let (best_trial_history, best_value_history) = best_value_history.into_iter().unzip();

    let x = seq(0, n_trial as i32 - 1, 1);

    let mut plt = Plot2D::new();
    plt.set_domain(x.clone())
        .insert_image(value_history)
        .insert_pair((best_trial_history, best_value_history))
        .set_plot_type(vec![(0, PlotType::Scatter), (1, PlotType::Line)])
        .set_marker(vec![(0, Markers::Point)])
        .set_color(vec![(0, "darkblue"), (1, "red")])
        .set_xlabel("Trial")
        .set_ylabel("Objective")
        .set_yscale(PlotScale::Log)
        .set_style(PlotStyle::Nature)
        .set_dpi(600)
        .set_path("optimize_history.png")
        .savefig()?;

    let mut plt = Plot2D::new();
    plt.set_domain(x.clone())
        .insert_image(x_history)
        .insert_image(y_history)
        .set_xlabel("Trial")
        .set_ylabel("Params")
        .set_plot_type(vec![(0, PlotType::Line), (1, PlotType::Scatter)])
        .set_marker(vec![(1, Markers::Point)])
        .set_legend(vec![r"$x$", r"$y$"])
        .set_color(vec![(0, "darkblue"), (1, "red")])
        .set_style(PlotStyle::Nature)
        .set_dpi(600)
        .set_path("parameter_history.png")
        .savefig()?;

    Ok(())
}

fn objective(x: f64, y: i32) -> f64 {
    x.powi(2) + y as f64
}
