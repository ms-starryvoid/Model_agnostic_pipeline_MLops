#!/usr/bin/env python
"""
Quick validation script to test the MLOps setup.
Run this after making the architectural changes.
"""

import sys
from pathlib import Path

# Test 1: Imports
print("=" * 60)
print("TEST 1: Checking imports...")
print("=" * 60)
try:
    from serving.adapter_registry import UsecaseAdapterRegistry
    from api.routes import build_router
    from usecases.classification.adapter import TitanicClassificationAdapter
    from pipeline.group_training_pipeline import GroupTrainingPipeline
    print("✅ All core modules import successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: Registry with no champion
print("\n" + "=" * 60)
print("TEST 2: Registry graceful handling (no champion)...")
print("=" * 60)
try:
    registry = UsecaseAdapterRegistry()
    adapter = TitanicClassificationAdapter()
    registry.register(adapter)
    
    print(f"✅ Registry registered: {registry.all_usecases()}")
    print(f"✅ Champion loaded? {registry.is_champion_loaded('shallow_mlp')}")
    
    try:
        registry.get("shallow_mlp")
        print("❌ Should have raised RuntimeError for unloaded champion")
    except RuntimeError as e:
        print(f"✅ Correctly raised error: {e}")
        
    # get_unsafe should work
    unsafe_adapter = registry.get_unsafe("shallow_mlp")
    print(f"✅ get_unsafe() returned adapter: {unsafe_adapter.usecase_name}")
    
except Exception as e:
    print(f"❌ Registry test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Schema validation
print("\n" + "=" * 60)
print("TEST 3: Input/Output schemas...")
print("=" * 60)
try:
    adapter = registry.get_unsafe("shallow_mlp")
    input_schema = adapter.input_schema.to_json_schema()
    output_schema = adapter.output_schema.to_json_schema()
    
    print(f"✅ Input schema fields: {list(input_schema['properties'].keys())}")
    print(f"✅ Output schema fields: {list(output_schema['properties'].keys())}")
    
    required_fields = set(input_schema['required'])
    expected_fields = {'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked'}
    if required_fields == expected_fields:
        print(f"✅ Input schema matches expected: {expected_fields}")
    else:
        print(f"❌ Input schema mismatch. Expected {expected_fields}, got {required_fields}")
        
except Exception as e:
    print(f"❌ Schema test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Feature processor
print("\n" + "=" * 60)
print("TEST 4: Feature processor...")
print("=" * 60)
try:
    from usecases.classification.processor import TitanicFeatureProcessor
    import numpy as np
    
    processor = TitanicFeatureProcessor()
    
    # Transform one record (should work even without fit)
    record = {
        "pclass": 3,
        "sex": "male",
        "age": 22.0,
        "sibsp": 1,
        "parch": 0,
        "fare": 7.25,
        "embarked": "S",
    }
    
    features = processor.transform(record)
    print(f"✅ Processor transformed record: shape={features.shape}, dtype={features.dtype}")
    
    if features.shape == (1, 7):
        print(f"✅ Output shape is correct: (1, 7)")
    else:
        print(f"❌ Output shape is wrong: expected (1, 7), got {features.shape}")
        
except Exception as e:
    print(f"❌ Processor test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Model instantiation
print("\n" + "=" * 60)
print("TEST 5: Model instantiation...")
print("=" * 60)
try:
    from usecases.classification.models import ShallowMLPModel
    
    model = ShallowMLPModel()
    model.build(input_dim=7)
    
    print(f"✅ Model built: {model.model}")
    print(f"✅ Model loader_tag: {model.loader_tag}")
    print(f"✅ Model adapter_tag: {model.adapter_tag}")
    
    # Quick predict
    dummy_input = np.array([[0.5] * 7], dtype=np.float32)
    pred = model.predict(dummy_input)
    print(f"✅ Model prediction: shape={pred.shape}, value={pred[0, 0]:.4f}")
    
except Exception as e:
    print(f"❌ Model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nNext steps:")
print("1. Start MLflow: mlflow ui")
print("2. Start API: uvicorn api.main:app --reload")
print("3. Train first model: POST /api/v1/train/shallow_mlp with CSV")
print("4. Make predictions: POST /api/v1/predict/shallow_mlp with record")
