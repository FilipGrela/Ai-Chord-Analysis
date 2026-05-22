import json
import os
from datetime import datetime
from dataclasses import asdict
from pathlib import Path

import optuna
import torch

from backend.config import cfg_hpo, cfg_model, cfg_paths, cfg_train
from backend.data.loader import DataLoaderFactory
from backend.logger.logger import Logger
from backend.models.crnn import ChordCRNN
from backend.training.loss import LossFactory
from backend.training.trainer import Trainer

logger = Logger(__name__)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    return torch.device("cpu")


def _backup_config() -> dict:
    return {
        "train": asdict(cfg_train),
        "model": asdict(cfg_model),
        "model_save_path": cfg_paths.MODEL_SAVE_PATH,
    }


def _restore_config(snapshot: dict) -> None:
    for key, value in snapshot["train"].items():
        setattr(cfg_train, key, value)
    for key, value in snapshot["model"].items():
        setattr(cfg_model, key, value)
    cfg_paths.MODEL_SAVE_PATH = snapshot["model_save_path"]


def _apply_trial_params(trial: optuna.Trial, model_out_dir: str) -> None:
    cfg_train.BATCH_SIZE = trial.suggest_categorical("batch_size", [32, 64, 128])
    cfg_train.LEARNING_RATE = trial.suggest_float("learning_rate", 3e-5, 3e-4, log=True)
    cfg_train.WEIGHT_DECAY = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    cfg_train.PATIENCE = trial.suggest_int("patience", 4, 8)



    cfg_model.DROPOUT_RATE = trial.suggest_float("dropout", 0.15, 0.35)
    cfg_model.RNN_HIDDEN_SIZE = trial.suggest_categorical("rnn_hidden_size", [128, 160, 192])
    cfg_model.RNN_NUM_LAYERS = trial.suggest_int("rnn_num_layers", 2, 4)

    cfg_train.AUGMENT_ENABLED = trial.suggest_categorical("augment_enabled", [True, False])

    cfg_train.AUGMENT_SPECMASK_ENABLED = trial.suggest_categorical("specmask_enabled", [True, False])

    cfg_train.AUGMENT_SPECMASK_PROB = trial.suggest_float("specmask_prob", 0.1, 0.5)
    cfg_train.AUGMENT_SPECMASK_MAX_TIME_MASKS = trial.suggest_int("specmask_max_time_masks", 1, 10)
    cfg_train.AUGMENT_SPECMASK_MAX_FREQ_MASKS = trial.suggest_int("specmask_max_freq_masks", 1, 10)
    cfg_train.AUGMENT_SPECMASK_MAX_TIME_WIDTH = trial.suggest_int("specmask_max_time_width", 1, 10)
    cfg_train.AUGMENT_SPECMASK_MAX_FREQ_WIDTH = trial.suggest_int("specmask_max_freq_width", 1, 10)

    cfg_paths.MODEL_SAVE_PATH = os.path.join(model_out_dir, "model.pth")


def _trial_dir(run_root: str, trial_number: int) -> str:
    # Struktura: out/hpo/run/trial1, out/hpo/run/trial2, ...
    return os.path.join(run_root, f"trial{trial_number + 1}")


def _write_trial_report(
    trial_dir: str,
    trial: optuna.Trial,
    result: dict | None,
    objective_value: float | None,
    status: str,
    metrics_dir: str | None = None,
    error_message: str | None = None,
) -> None:
    report_path = os.path.join(trial_dir, "report.txt")
    lines = [
        f"trial_number: {trial.number}",
        f"status: {status}",
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"objective(best_val_loss): {objective_value if objective_value is not None else 'N/A'}",
        "",
        "params:",
    ]

    for key, value in sorted(trial.params.items()):
        lines.append(f"  {key}: {value}")

    lines.extend(["", "metrics:"])
    if result is not None:
        for key in ["best_val_loss", "best_val_acc", "best_epoch", "epochs_trained"]:
            lines.append(f"  {key}: {result.get(key)}")
    else:
        lines.append("  N/A")

    if error_message:
        lines.extend(["", "error:", f"  {error_message}"])

    if metrics_dir:
        lines.extend(["", "generated_files:"])
        metrics_path = Path(metrics_dir)
        if metrics_path.exists():
            generated = sorted(p.name for p in metrics_path.iterdir() if p.is_file())
            if generated:
                for name in generated:
                    lines.append(f"  metrics/{name}")
            else:
                lines.append("  metrics/(no files)")
        else:
            lines.append("  metrics/(missing directory)")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _build_objective(device: torch.device, run_root: str):
    def objective(trial: optuna.Trial) -> float:
        snapshot = _backup_config()
        trial_dir = _trial_dir(run_root, trial.number)
        os.makedirs(trial_dir, exist_ok=True)
        metrics_dir = os.path.join(trial_dir, "metrics")

        result = None
        try:
            _apply_trial_params(trial, trial_dir)

            train_loader, val_loader = DataLoaderFactory.create_dataloaders(
                data_dir=cfg_paths.PROCESSED_DATA,
                batch_size=cfg_train.BATCH_SIZE,
            )
            model = ChordCRNN()
            criterion = LossFactory.create_loss_function(train_loader, device)

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                enable_epoch_metrics=True,
                metrics_output_dir=metrics_dir,
            )

            result = trainer.train()
            trial.set_user_attr("best_val_acc", result["best_val_acc"])
            trial.set_user_attr("best_epoch", result["best_epoch"])
            trial.set_user_attr("epochs_trained", result["epochs_trained"])

            objective_value = result["best_val_loss"]
            _write_trial_report(
                trial_dir=trial_dir,
                trial=trial,
                result=result,
                objective_value=objective_value,
                status="completed",
                metrics_dir=metrics_dir,
            )
            return objective_value
        except KeyboardInterrupt:
            # User aborted the trial; write an 'aborted' report and re-raise to stop HPO
            _write_trial_report(
                trial_dir=trial_dir,
                trial=trial,
                result=result,
                objective_value=None,
                status="aborted",
                metrics_dir=metrics_dir,
                error_message="KeyboardInterrupt (user aborted)",
            )
            raise
        except Exception as exc:
            _write_trial_report(
                trial_dir=trial_dir,
                trial=trial,
                result=result,
                objective_value=None,
                status="failed",
                metrics_dir=metrics_dir,
                error_message=str(exc),
            )
            raise
        finally:
            _restore_config(snapshot)

    return objective


def main() -> None:
    os.makedirs(cfg_hpo.OUTPUT_DIR, exist_ok=True)
    run_root = os.path.join(cfg_hpo.OUTPUT_DIR, "run")
    os.makedirs(run_root, exist_ok=True)

    device = _get_device()
    logger.info(f"HPO device: {device}")
    logger.info(
        f"Start HPO | study={cfg_hpo.STUDY_NAME} | trials={cfg_hpo.N_TRIALS} | timeout={cfg_hpo.TIMEOUT_SECONDS}"
    )

    storage = f"sqlite:///{cfg_hpo.STORAGE_PATH}"
    os.makedirs(os.path.dirname(cfg_hpo.STORAGE_PATH), exist_ok=True)

    study = optuna.create_study(
        study_name=cfg_hpo.STUDY_NAME,
        direction=cfg_hpo.DIRECTION,
        storage=storage,
        load_if_exists=True,
    )

    objective = _build_objective(device, run_root)
    try:
        study.optimize(objective, n_trials=cfg_hpo.N_TRIALS, timeout=cfg_hpo.TIMEOUT_SECONDS)
    except KeyboardInterrupt:
        logger.warning("HPO przerwane przez użytkownika (KeyboardInterrupt). Zapisywanie dotychczasowych wyników...")
        try:
            # zapisz aktualne najlepsze parametry (jeśli istnieją)
            if study.best_trial is not None:
                best = {
                    "best_value": study.best_value,
                    "best_params": study.best_params,
                    "best_trial_number": study.best_trial.number,
                    "best_trial_user_attrs": getattr(study.best_trial, "user_attrs", {}),
                }
                best_path = os.path.join(cfg_hpo.OUTPUT_DIR, "best_params_interrupted.json")
                with open(best_path, "w", encoding="utf-8") as f:
                    json.dump(best, f, indent=2)
                logger.info(f"Zapisano najlepsze parametry (przerwane): {best_path}")
        except Exception as exc:
            logger.error(f"Nie udało się zapisać najlepszych parametrów po przerwaniu: {exc}")
        raise

    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "best_trial_user_attrs": study.best_trial.user_attrs,
    }

    best_path = os.path.join(cfg_hpo.OUTPUT_DIR, "best_params.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    best_txt_path = os.path.join(cfg_hpo.OUTPUT_DIR, "best_params.txt")
    with open(best_txt_path, "w", encoding="utf-8") as f:
        f.write(f"best_value: {best['best_value']}\n")
        f.write(f"best_trial_number: {best['best_trial_number']}\n")
        f.write("best_params:\n")
        for key, value in sorted(best["best_params"].items()):
            f.write(f"  {key}: {value}\n")
        f.write("best_trial_user_attrs:\n")
        for key, value in sorted(best["best_trial_user_attrs"].items()):
            f.write(f"  {key}: {value}\n")

    logger.info(f"HPO finished. Best val_loss={study.best_value:.6f}")
    logger.info(f"Best params saved to: {best_path}")


if __name__ == "__main__":
    main()
