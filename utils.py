from pathlib import Path
from datetime import datetime

# Formula
formula = ["(F c0) & (F c1)", "2", "task0: visit(gem, door)"]

# Output file structure.
out_folder = "data/"
model_folder = out_folder + "model/"
log_folder = out_folder + "log/"
eval_folder = out_folder + "eval/"
plot_folder = out_folder + "plot/"

def ensure_directories(formula_name: str):
    """Ensure that the necessary directories for a given formula exist."""
    experiment_folder = formula_name.replace(" ", "_") + "_" + datetime.now().strftime("%Y-%m-%d.%H:%M:%S") + "/"
    global out_folder, model_folder, log_folder, eval_folder, plot_folder
    model_folder = model_folder + experiment_folder
    log_folder = log_folder + experiment_folder
    eval_folder = eval_folder + experiment_folder
    plot_folder = plot_folder + experiment_folder
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    Path(eval_folder).mkdir(parents=True, exist_ok=True)
    Path(plot_folder).mkdir(parents=True, exist_ok=True)

def get_plot_folder(log_file: str) -> str:
    """Get the plot folder path corresponding to a given log file."""
    log_path = Path(log_file)
    plot_path = log_path.parent.parent.parent / "plot" / log_path.parent.name
    plot_path.mkdir(parents=True, exist_ok=True)
    return str(plot_path) + "/"

def get_eval_folder(model_file: str) -> str:
    """Get the eval folder path corresponding to a given model file."""
    model_path = Path(model_file)
    eval_path = model_path.parent.parent.parent / "eval" / model_path.parent.name
    eval_path.mkdir(parents=True, exist_ok=True)
    return str(eval_path) + "/"