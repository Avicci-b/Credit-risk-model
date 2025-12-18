"""
Run script for Credit Risk Model API
"""

import subprocess
import sys
import os
import time

def check_dependencies():
    """Check if all required packages are installed"""
    required = ['fastapi', 'uvicorn', 'pydantic']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install fastapi uvicorn pydantic")
        return False
    
    return True

def check_model_files():
    """Check if model files exist"""
    required_files = [
        'models/best_model.joblib',
        'models/label_encoders.joblib',
        'models/scaler.joblib',
        'models/feature_names.json'
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    if missing:
        print(f"⚠️  Missing model files: {', '.join(missing)}")
        print("Creating dummy model files for testing...")
        
        # Create dummy model files
        os.makedirs('models', exist_ok=True)
        
        import joblib
        import json
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        # Create dummy model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_dummy = np.random.randn(10, 5)
        y_dummy = np.random.randint(0, 2, 10)
        model.fit(X_dummy, y_dummy)
        
        # Save dummy artifacts
        joblib.dump(model, 'models/best_model.joblib')
        joblib.dump({'CurrencyCode': 'encoder', 'ProductCategory': 'encoder'}, 
                   'models/label_encoders.joblib')
        joblib.dump('scaler', 'models/scaler.joblib')
        
        with open('models/feature_names.json', 'w') as f:
            json.dump(['feature1', 'feature2', 'feature3', 'feature4', 'feature5'], f)
        
        print("✅ Created dummy model files for testing")
    
    return True

def main():
    """Run the API"""
    print("=" * 60)
    print("CREDIT RISK MODEL API")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check model files
    if not check_model_files():
        return 1
    
    print("\nStarting API server...")
    print("API will be available at: http://localhost:8000")
    print("Documentation: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        # Start the API server
        proc = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "src.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
        
        # Wait for process
        proc.wait()
        
    except KeyboardInterrupt:
        print("\n\nStopping API server...")
        proc.terminate()
        proc.wait()
        print("✅ API server stopped")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())