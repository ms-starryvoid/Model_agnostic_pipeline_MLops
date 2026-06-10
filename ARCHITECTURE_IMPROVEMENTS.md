# MLOps Architecture: Changes & Next Steps

## Problem Statement
Your project had **three critical issues**:

1. **API crashes on startup** if no champion model exists
   - `registry.register()` calls `load_champion()` eagerly
   - FileNotFoundError when processor.joblib doesn't exist

2. **Processor artifact not bundled with model**
   - `ModelContract.log_to_mlflow()` only logs weights, not processor
   - Adapter tries to load processor.joblib from temp directory → fails

3. **No bootstrap path** to train the first model
   - No way to train models without manually managing MLflow artifacts

---

## Solutions Implemented ✅

### 1. **Graceful Initialization** 
**File:** `serving/adapter_registry.py`

```python
def register(self, adapter: UsecaseAdapter) -> None:
    """Register adapter. Champion loading is deferred + optional."""
    self._adapters[adapter.usecase_name] = adapter
    try:
        adapter.load_champion()
    except Exception as e:
        logger.warning(f"No champion found for '{adapter.usecase_name}' — "
                       f"API will accept /train requests. Error: {e}")
```

**Key changes:**
- ✅ API starts successfully even if no champion exists
- ✅ `is_champion_loaded()` tracks which usecases have working models
- ✅ `get()` raises RuntimeError if champion not loaded (for /predict)
- ✅ `get_unsafe()` returns adapter without champion check (for /train)
- ✅ `hot_swap()` is now fault-tolerant and logs errors

**Effect:** API starts, lists usecases as "no_champion", and waits for training.

---

### 2. **Processor Bundling** 
**Files:** 
- `contracts/model_contract.py` (accept processor parameter)
- `usecases/classification/models.py` (log processor with joblib)
- `pipeline/group_training_pipeline.py` (pass processor to log_to_mlflow)

```python
# In ShallowMLPModel.log_to_mlflow()
def log_to_mlflow(self, run: mlflow.ActiveRun, artifact_path: str, 
                  processor=None, **kwargs) -> str:
    # Log model weights
    mlflow.pytorch.log_model(pytorch_model=scripted_model, ...)
    
    # CRITICAL: Log processor
    if processor is not None:
        processor_path = f"{artifact_path}/processor.joblib"
        joblib.dump(processor, processor_path)
        mlflow.log_artifact(processor_path)
```

**In GroupTrainingPipeline:**
```python
uri = model.log_to_mlflow(run, artifact_path="model", processor=processor)
```

**Effect:** Both model weights and processor are now in the same MLflow artifact directory.

---

### 3. **Training Endpoint** 
**File:** `api/routes.py`

```python
@router.post("/train/{usecase}")
async def train_usecase(usecase: str, csv_file: UploadFile = File(...), 
                        target_col: str = "target"):
    """
    Train + register a champion model for a usecase.
    Expects CSV with training data.
    """
    # Read CSV, validate target_col exists
    # Instantiate group, processor, models
    # Run GroupTrainingPipeline
    # Hot-swap champion into registry
```

**Effect:** You can now bootstrap models via API without external MLflow setup.

---

## Quick Start: Train Your First Model

### **Scenario 1: API running, no champion yet**

```bash
# Terminal 1: Start API
uvicorn api.main:app --reload

# Terminal 2: Check status
curl http://localhost:8000/api/v1/usecases
# Response: { "usecases": [{ "name": "shallow_mlp", "status": "no_champion", ... }] }

# Terminal 3: Upload Titanic CSV to train
curl -X POST \
  -F "csv_file=@titanic.csv" \
  -F "target_col=survived" \
  http://localhost:8000/api/v1/train/shallow_mlp

# Response: { "message": "Training completed successfully", "champion_loaded": true }

# Now predictions work
curl -X POST http://localhost:8000/api/v1/predict/shallow_mlp \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 3,
    "sex": "male", 
    "age": 22.0,
    "sibsp": 1,
    "parch": 0,
    "fare": 7.25,
    "embarked": "S"
  }'
```

### **Scenario 2: Train via CLI (for batch jobs)**

Create `train_titanic.py`:

```python
import pandas as pd
from usecases.classification.groups import TitanicClassificationGroup
from usecases.classification.processor import TitanicFeatureProcessor
from usecases.classification.models import ShallowMLPModel
from pipeline.group_training_pipeline import GroupTrainingPipeline

group = TitanicClassificationGroup()
processor = TitanicFeatureProcessor()
models = [ShallowMLPModel()]

df = pd.read_csv("titanic.csv")
pipeline = GroupTrainingPipeline(group, experiment_name="titanic_batch")
best_uri = pipeline.run(
    models=models,
    processor=processor,
    raw_df=df,
    target_col="survived",
    metric_to_maximize="loss"
)
print(f"Champion trained: {best_uri}")
```

Run: `python train_titanic.py`

---

## Architecture Alignment with Your Requirements

### ✅ **What you asked for:**

| Requirement | Status | Where |
|------------|--------|-------|
| Groups of models for scenarios | ✅ Yes | `UsecaseGroupContract` + `GroupTrainingPipeline` |
| Champion model with tags | ✅ Yes | MLflow aliases + `mlflow_tags` property |
| Generic feature processing | ✅ Yes | `FeatureContract` (e.g., `TitanicFeatureProcessor`) |
| Strict input/output shapes | ✅ Yes | `InputSchema` + `OutputSchema` with validation |
| Single adapter for preprocessing | ✅ Yes | `UsecaseAdapter.preprocess()` at inference |
| API endpoints for schema + champion | ✅ Yes | `/usecases`, `/schema/{usecase}`, `/predict/{usecase}` |
| Loose coupling | ✅ Yes | Abstract contracts, adapter registry pattern |

### ⚠️ **Still to implement (next level):**

1. **Parameterized training** - Currently hardcoded to Titanic classification
   - Next: Read group + model class from query params or config

2. **Multiple model variants** - GroupTrainingPipeline accepts `list[ModelContract]`
   - Currently only trains ShallowMLPModel
   - Next: Add XGBoost, SklearnModel variants, compare automatically

3. **Model evaluation metrics** - Currently just logs loss
   - Next: F1, precision, recall, ROC-AUC; log confusion matrix

4. **A/B testing framework** - Champion is latest best model
   - Next: Candidate alias, shadow traffic routing, statistical significance

5. **Data/model drift monitoring** - ChampionWatcher only hot-swaps
   - Next: Monitor input distributions, model performance decay

6. **Feature store integration** - Features computed ad-hoc in adapter
   - Next: Fetch pre-computed features from feature store (e.g., Feast, Hopsworks)

---

## File-by-File Changes Summary

| File | Change | Why |
|------|--------|-----|
| `serving/adapter_registry.py` | Defer + fault-tolerant champion loading | API starts safely |
| `api/routes.py` | Add status checks, new `/train/{usecase}` endpoint | Bootstrap & monitor |
| `contracts/model_contract.py` | Accept processor parameter in `log_to_mlflow()` | Bundle artifacts |
| `usecases/classification/models.py` | Log processor with joblib | Processor available at inference |
| `pipeline/group_training_pipeline.py` | Pass processor to `log_to_mlflow()` | Processor bundled |
| `usecases/classification/adapter.py` | Improve processor loading paths + error messages | Robust artifact handling |

---

## Testing Your Setup

### **Before running, ensure:**
1. MLflow server is accessible (check `config.py` → `mlflow_tracking_uri`)
2. Titanic CSV exists with columns: `pclass, sex, age, sibsp, parch, fare, embarked, survived`

### **Run this to verify:**

```python
# test_setup.py
from registry import build_registry
from config import settings

print(f"MLflow URI: {settings.mlflow_tracking_uri}")

try:
    registry = build_registry()
    print(f"Registered usecases: {registry.all_usecases()}")
    for uc in registry.all_usecases():
        loaded = registry.is_champion_loaded(uc)
        print(f"  - {uc}: {'✅ Champion loaded' if loaded else '⚠️ No champion (use /train)'}")
except Exception as e:
    print(f"Error: {e}")
```

---

## Debugging Tips

**API starts but /predict fails:**
```
RuntimeError: Champion not yet loaded for 'shallow_mlp'. 
POST /api/v1/train/shallow_mlp to train first.
```
→ Use the `/train/{usecase}` endpoint with your CSV.

**Training endpoint fails with validation error:**
```
422: Target column 'target' not found in CSV. Available: ['pclass', 'sex', ...]
```
→ Pass correct `target_col` as query param: `/train/shallow_mlp?target_col=survived`

**processor.joblib not found during training:**
→ Ensure `ShallowMLPModel.log_to_mlflow()` is receiving the `processor` parameter (it is, via `GroupTrainingPipeline`).

---

## Next Steps Recommendation

1. **Phase 1 (Foundation):** Test current setup with training endpoint
2. **Phase 2 (Flexibility):** Parameterize which group/models to train
3. **Phase 3 (Monitoring):** Add model performance metrics + data drift
4. **Phase 4 (Production):** A/B testing, shadow routing, feature store
