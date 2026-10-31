import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.data import load_params
from src.train import train

TRACKING_URI = "http://127.0.0.1:5000"

params = load_params()
gate = params["quality_gate"]
model_name = params["training"]["model_name"]

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(params["training"]["experiment_name"])
client = MlflowClient()


def get_baseline_metrics():
    """Métriques du modèle champion actuel (None s'il n'y en a pas encore)."""
    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
    except MlflowException:
        return None
    run = client.get_run(champion.run_id)
    return run.data.metrics


with mlflow.start_run(run_name="sentiment-candidate"):
    model, metrics = train()

    mlflow.log_params({
        "C": params["model"]["C"],
        "max_features": params["model"]["max_features"],
        "ngram_range": str(params["model"]["ngram_range"]),
    })
    mlflow.log_metrics(metrics)

    # --- Quality gates : seuils absolus (G3, G4) ---
    checks = {
        "G3 accuracy >= seuil": metrics["accuracy"] >= gate["min_accuracy"],
        "G4 f1 >= seuil": metrics["f1"] >= gate["min_f1"],
    }

    # --- Non-régression : candidat >= baseline - delta ---
    baseline = get_baseline_metrics()
    if baseline is not None:
        delta = gate["max_regression"]
        for name in ("accuracy", "f1"):
            checks[f"non-regression {name}"] = (
                metrics[name] >= baseline[name] - delta
            )
    else:
        print("Pas de champion existant : pas de comparaison à une baseline.")

    for label, ok in checks.items():
        print(f"{label} : {'OK' if ok else 'ECHEC'}")

    status = "passed" if all(checks.values()) else "failed"

    model_info = mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name=model_name,
    )

versions = client.search_model_versions(f"name = '{model_name}'")
latest = max(versions, key=lambda v: int(v.version))

client.set_model_version_tag(model_name, latest.version, "validation_status", status)
client.set_model_version_tag(model_name, latest.version, "use_case", "sentiment")
client.set_registered_model_tag(model_name, "task", "text-classification")
client.set_registered_model_alias(model_name, "candidate", latest.version)

print(f"Version enregistree : {latest.version}")
print(f"Alias candidate -> {latest.version}")
print(f"validation_status = {status}")
print(f"Accuracy : {metrics['accuracy']:.3f}")
print(f"F1 : {metrics['f1']:.3f}")
