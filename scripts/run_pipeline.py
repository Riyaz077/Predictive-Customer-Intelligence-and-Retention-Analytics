from scripts.generate_data import generate_customers
from src.customer360.pipeline import run_pipeline
from pathlib import Path

if __name__ == "__main__":
    data = Path("data/sample/customers.csv")
    if not data.exists():
        data.parent.mkdir(parents=True, exist_ok=True)
        generate_customers().to_csv(data, index=False)
    metrics, _ = run_pipeline(data_path=str(data))
    print("Training complete. Metrics:")
    for task, values in metrics.items():
        print(task, values)
