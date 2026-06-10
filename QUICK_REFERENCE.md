# 🚀 Quick Reference: Architecture Changes

## What Was Fixed

| Problem | Solution | File |
|---------|----------|------|
| 🔴 API crashes if no champion | Graceful initialization with fallback | `adapter_registry.py` |
| 🔴 Processor.joblib not logged | Bundle processor with model artifact | `models.py`, `group_training_pipeline.py` |
| 🔴 No way to train first model | New POST `/train/{usecase}` endpoint | `routes.py` |
| 🔴 Robustness issues on load | Better error handling + logging | `adapter.py` |

---

## 🎯 New API Endpoints

### 1. Check Usecase Status
```bash
GET /api/v1/usecases
```
Response:
```json
{
  "usecases": [
    {
      "name": "shallow_mlp",
      "status": "no_champion",
      "input_schema": { ... },
      "output_schema": { ... },
      "message": "No champion loaded. POST /api/v1/train/shallow_mlp to train."
    }
  ]
}
```

### 2. Get Input/Output Schema
```bash
GET /api/v1/schema/shallow_mlp
```

### 3. **[NEW]** Train & Register Champion
```bash
POST /api/v1/train/shallow_mlp
Content-Type: multipart/form-data

csv_file: @titanic.csv
target_col: survived (optional, defaults to "target")
```
Response:
```json
{
  "usecase": "shallow_mlp",
  "message": "Training completed successfully",
  "best_model_uri": "models:/shallow_mlp_models/champion",
  "champion_loaded": true
}
```

### 4. Make Prediction (requires champion)
```bash
POST /api/v1/predict/shallow_mlp
Content-Type: application/json

{
  "pclass": 3,
  "sex": "male",
  "age": 22.0,
  "sibsp": 1,
  "parch": 0,
  "fare": 7.25,
  "embarked": "S"
}
```
Response:
```json
{
  "usecase": "shallow_mlp",
  "prediction": {
    "survived": true,
    "probability": 0.7234
  }
}
```

---

## 🏗️ Architecture Alignment Scorecard

| Requirement | Status | Notes |
|-------------|--------|-------|
| Groups of models | ✅ | `UsecaseGroupContract` + list of `ModelContract` |
| Champion selection | ✅ | MLflow aliases (registry.register_model_alias) |
| Generic feature processing | ✅ | `FeatureContract` interface |
| Strict input/output schemas | ✅ | `InputSchema` + `OutputSchema` with JSON schema export |
| Single preprocessing adapter | ✅ | `UsecaseAdapter.preprocess()` at inference |
| Loose coupling | ✅ | Abstract contracts + dependency injection |
| CI/CD ready | ✅ | Training pipeline, hot-swap, graceful init |

---

## 📋 Testing Checklist

- [ ] MLflow server running (check `config.py` for URI)
- [ ] Start API: `uvicorn api.main:app --reload`
- [ ] Check status: `GET /api/v1/usecases` → status should be "no_champion"
- [ ] Upload CSV: `POST /api/v1/train/shallow_mlp` with your titanic.csv
- [ ] Verify champion loaded: `GET /api/v1/usecases` → status should be "champion_loaded"
- [ ] Make prediction: `POST /api/v1/predict/shallow_mlp` with sample record
- [ ] Verify hot-swap: Train again, ChampionWatcher should reload (30s interval)

---

## 🔧 Development Workflow

### **Scenario 1: Fresh API (no models yet)**
```bash
# 1. Ensure MLflow is running
mlflow ui  # http://localhost:5000

# 2. Start FastAPI
cd /path/to/project
uvicorn api.main:app --reload

# 3. Train first champion
curl -X POST -F "csv_file=@data/titanic.csv" \
  http://localhost:8000/api/v1/train/shallow_mlp

# 4. Make predictions
curl -X POST http://localhost:8000/api/v1/predict/shallow_mlp \
  -H "Content-Type: application/json" \
  -d '{"pclass": 3, "sex": "male", "age": 22, "sibsp": 1, "parch": 0, "fare": 7.25, "embarked": "S"}'
```

### **Scenario 2: Retrain with new data**
```bash
# New data with better metrics
curl -X POST -F "csv_file=@data/titanic_v2.csv" \
  http://localhost:8000/api/v1/train/shallow_mlp

# ChampionWatcher will auto-reload if F1 improves
# (30 second polling interval by default)
```

### **Scenario 3: Batch training (CI/CD)**
```python
# scripts/train_batch.py
from pipeline.group_training_pipeline import GroupTrainingPipeline
from usecases.classification.groups import TitanicClassificationGroup
from usecases.classification.processor import TitanicFeatureProcessor
from usecases.classification.models import ShallowMLPModel
import pandas as pd

group = TitanicClassificationGroup()
processor = TitanicFeatureProcessor()
models = [ShallowMLPModel()]

df = pd.read_csv("data/titanic_production.csv")
pipeline = GroupTrainingPipeline(group)
uri = pipeline.run(models, processor, df, "survived")
print(f"Champion trained: {uri}")
```

---

## ⚙️ Configuration

**File:** `config.py`
```python
class Settings(BaseSettings):
    mlflow_tracking_uri: str = "http://localhost:5000"  # MLflow server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    champion_poll_interval: int = 30  # seconds
```

**Set via env or .env file:**
```bash
MLFLOW_TRACKING_URI=http://localhost:5000
API_HOST=0.0.0.0
API_PORT=8000
```

---

## 🐛 Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: processor.joblib` | Old models logged before bundling | Retrain with new code |
| `RuntimeError: Champion not yet loaded` | No model trained yet | POST `/train/{usecase}` |
| `HTTPException 422: Target column not found` | CSV missing target column | Verify column name, pass as `target_col` param |
| `MlflowException: No champion found` | Model registered but no alias | Manually set alias or retrain |
| `ConnectionError: MLflow server` | MLflow not running | Start: `mlflow ui` |

---

## 📚 Key Files Modified

```
serving/adapter_registry.py        ← Graceful init, champion loading logic
api/routes.py                      ← New /train endpoint
contracts/model_contract.py        ← Accept processor parameter
usecases/classification/
  ├── models.py                    ← Log processor artifact
  └── adapter.py                   ← Robust artifact loading
pipeline/group_training_pipeline.py ← Pass processor to log_to_mlflow()
```

---

## 🚀 Next Level Features (Future)

1. **Parameterized Training** → Train any model group via API
2. **Multiple Variants** → Compare XGBoost vs Neural Net automatically
3. **A/B Testing** → Candidate alias, shadow traffic routing
4. **Monitoring** → Data/model drift detection
5. **Feature Store** → Fetch pre-computed features from Feast/Hopsworks
6. **Batch Predictions** → Process CSV files at inference time

