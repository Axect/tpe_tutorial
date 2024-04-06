use peroxide::{fuga::*, hstack};
use rand::SeedableRng;
use tpe::{parzen_estimator, range, TpeOptimizer, TpeOptimizerBuilder};
use indicatif::{ProgressBar, ProgressStyle};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let noise = Normal(0., 0.5);
    let p_true = vec![20f64, 10f64, 50f64];
    let p_init = vec![5f64, 2f64, 10f64];

    let domain = linspace(0, 100, 1000);
    let y_model = model(&domain, &p_true);
    let y_true = zip_with(|x, y| x + y, &y_model, &noise.sample(domain.len())); 
    let data = hstack!(domain.clone(), y_true.clone());

    // LM
    let mut lm_optim = Optimizer::new(data, f);
    let best_param_lm = lm_optim
        .set_init_param(p_init.clone())
        .set_max_iter(100)
        .set_method(LevenbergMarquardt)
        .optimize();
    best_param_lm.print();
    lm_optim.get_error().print();

    // TPE
    let mut tpe_optim = vec![];
    for _ in 0 .. p_true.len() {
        let mut tpe_optim_builder = TpeOptimizerBuilder::new();
        tpe_optim_builder.gamma(0.2);
        tpe_optim_builder.candidates(10);
        let tpe_optimizer = tpe_optim_builder.build(parzen_estimator(), range(0.0, 100.0)?)?;
        tpe_optim.push(tpe_optimizer);
    }

    let n_trial = 1000;

    let mut best_params = p_init.clone();
    let mut best_trial = 0usize;
    let mut best_value = std::f64::INFINITY;
    let mut value_history = vec![0f64; n_trial];
    let mut param_history = vec![vec![0f64; p_init.len()]; n_trial];
    let mut best_value_history = vec![];

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let progress_bar = ProgressBar::new(n_trial as u64);
    progress_bar.set_style(ProgressStyle::default_bar());
    for i in 0..n_trial {
        progress_bar.inc(1);
        let mut p_tpe = vec![0f64; p_init.len()];
        for i in 0 .. p_init.len() {
            p_tpe[i] = tpe_optim[i].ask(&mut rng)?;
        }

        let y_tpe = model(&domain, &p_tpe);
        let obj_tpe = objective(&y_true, &y_tpe);
        for i in 0 .. p_init.len() {
            tpe_optim[i].tell(p_tpe[i], obj_tpe)?;
        }

        if obj_tpe < best_value {
            best_params.copy_from_slice(&p_tpe);
            best_trial = i;
            best_value = obj_tpe;
            best_value_history.push((i as f64, obj_tpe));
        }

        value_history[i] = obj_tpe;
        param_history[i] = p_tpe;
    }

    println!("Best params: {:?}", best_params);
    println!("Best trial: {}", best_trial);
    println!("Best value: {}", best_value);

    let param_history = py_matrix(param_history);
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

    let line_style_cand = vec![LineStyle::Solid, LineStyle::Dashed, LineStyle::Dotted, LineStyle::DashDot];
    let line_style = (0 .. p_init.len()).map(|i| (i, line_style_cand[i % line_style_cand.len()])).collect();
    let mut plt = Plot2D::new();
    plt.set_domain(x.clone());
    for i in 0..p_init.len() {
        plt.insert_image(param_history.col(i));
    }
    plt
        .set_xlabel("Trial")
        .set_ylabel("Params")
        .set_line_style(line_style)
        .set_legend(vec![r"$p_0$", r"$p_1$", r"$p_2$"])
        .set_style(PlotStyle::Nature)
        .set_dpi(600)
        .set_path("parameter_history.png")
        .savefig()?;


    // LM vs TPE
    let mut plt = Plot2D::new();
    plt
        .set_domain(domain.clone())
        .insert_image(y_true)
        .insert_image(y_model)
        .insert_image(model(&domain, &best_param_lm))
        .insert_image(model(&domain, &best_params))
        .set_plot_type(vec![(0, PlotType::Scatter), (1, PlotType::Line), (2, PlotType::Line), (3, PlotType::Line)])
        .set_marker(vec![(0, Markers::Point)])
        .set_line_style(vec![(1, LineStyle::Solid), (2, LineStyle::Dashed), (3, LineStyle::Dotted)])
        .set_color(vec![(0, "blue"), (1, "black"), (2, "red"), (3, "green")])
        .set_alpha(vec![(0, 0.2), (1, 0.7), (2, 0.6), (3, 0.6)])
        .set_xlabel("x")
        .set_ylabel("y")
        .set_legend(vec!["Data", "True", "LM", "TPE"])
        .set_style(PlotStyle::Nature)
        .tight_layout()
        .set_dpi(600)
        .set_path("lm_tpe.png")
        .savefig()?;

    Ok(())
}

fn model(x: &[f64], p: &[f64]) -> Vec<f64> {
    x.iter().map(|&t| {
        p[0] * (-t / p[1]).exp() + t * (-t / p[2]).exp()
    }).collect()
}

fn objective(y: &[f64], y_hat: &[f64]) -> f64 {
    let error: Vec<f64> = y.iter().zip(y_hat).map(|(&y, &y_hat)| (y - y_hat).abs()).collect();
    error.mean()
}

fn f(domain: &Vec<f64>, p: Vec<AD>) -> Option<Vec<AD>> {
    Some(
        domain.clone().into_iter()
            .map(|t| AD1(t, 0f64))
            .map(|t| p[0] * (-t / p[1]).exp() + t * (-t / p[2]).exp())
            .collect()
    )
}
