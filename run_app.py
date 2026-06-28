"""
Spec2Prop-Edge Launcher
=======================
Starts the FastAPI backend and Vite frontend together.
"""
import subprocess
import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=======================================")
    print(" Starting Spec2Prop-Edge")
    print("=======================================")
    
    # 1. Start FastAPI Backend
    backend_cmd = [sys.executable, "-m", "uvicorn", "app.backend.main:app", "--reload", "--port", "8000"]
    print(f"Starting backend: {' '.join(backend_cmd)}")
    backend_proc = subprocess.Popen(backend_cmd, cwd=PROJECT_ROOT)
    
    # Give it a second to boot
    time.sleep(2)
    
    # 2. Start Vite Frontend
    frontend_dir = os.path.join(PROJECT_ROOT, "app", "frontend")
    
    # Check if node_modules exists, if not run npm install
    if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
        print("Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, shell=True)
        
    frontend_cmd = ["npm", "run", "dev"]
    print(f"Starting frontend: {' '.join(frontend_cmd)}")
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=frontend_dir, shell=True)
    
    try:
        print("\nApp is running! Press Ctrl+C to stop.")
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend_proc.terminate()
        frontend_proc.terminate()
        sys.exit(0)

if __name__ == "__main__":
    main()
