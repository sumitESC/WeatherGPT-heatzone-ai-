import os
import shutil
import subprocess
from datetime import datetime

# GitHub repository details
REPO_NAME = "sumitESC/W-backend"
BRANCH = "main" # Change to 'master' if your default branch is master

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# List of local files to push to W-backend (paths relative to this script)
FILES_TO_PUSH = [
    "data/all_cities_heatscore_forecast.json",
    "data/processed/india_regional_weather.csv",
    "data/processed/ml_ready_historical_data.csv",
    "data/processed/india_context_weather.csv",
    "data/processed/ml_ready_context_data.csv",
    "render.yaml"
]

def push_data_to_github():
    # Use standard HTTPS URL. Git will automatically use saved credentials (e.g., Windows Credential Manager)
    repo_url = f"https://github.com/sumitESC/W-backend.git"
    
    # Use a unique timestamped directory to avoid lock conflicts
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_dir = f"temp_w_backend_repo_{timestamp_str}"
    
    def remove_readonly(func, path, excinfo):
        import stat
        os.chmod(path, stat.S_IWRITE)
        func(path)

    try:
        # 1. Clone the repository
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, onerror=remove_readonly)
            
        print(f"Cloning {REPO_NAME}...")
        subprocess.run(["git", "clone", "--branch", BRANCH, repo_url, temp_dir], check=True, capture_output=True)

        # 2. Copy the specified files into the cloned repository
        print("Copying files...")
        files_copied = 0
        for rel_file_path in FILES_TO_PUSH:
            abs_file_path = os.path.join(SCRIPT_DIR, rel_file_path)
            
            if os.path.exists(abs_file_path):
                # Create directories if they don't exist in the target
                target_path = os.path.join(temp_dir, rel_file_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                shutil.copy2(abs_file_path, target_path)
                print(f"  - Copied {rel_file_path}")
                files_copied += 1
            else:
                print(f"  - Warning: Local file {abs_file_path} not found. Skipping.")

        if files_copied == 0:
            print("No files were found to push. Exiting.")
            return

        # 3. Add, Commit, and Push
        commit_message = f"Update forecast and training datasets - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        print("Committing and pushing to GitHub...")
        subprocess.run(["git", "config", "user.name", "Automated Bot"], cwd=temp_dir, check=True)
        subprocess.run(["git", "config", "user.email", "bot@heatzone.local"], cwd=temp_dir, check=True)
        
        subprocess.run(["git", "add", "."], cwd=temp_dir, check=True)
        
        # Check if there are changes to commit
        status = subprocess.run(["git", "status", "--porcelain"], cwd=temp_dir, capture_output=True, text=True)
        if not status.stdout.strip():
            print("No changes to commit. Data is already up to date.")
            return

        subprocess.run(["git", "commit", "-m", commit_message], cwd=temp_dir, check=True, capture_output=True)
        subprocess.run(["git", "push", "origin", BRANCH], cwd=temp_dir, check=True, capture_output=True)
        
        print(f"Success! Committed {files_copied} files to {REPO_NAME}")

    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr.decode('utf-8', errors='ignore')}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # 4. Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, onerror=remove_readonly)
            print("Cleaned up temporary files.")

        # Also try to clean up the previous stuck temp dir if it exists
        if os.path.exists("temp_w_backend_repo"):
            try:
                shutil.rmtree("temp_w_backend_repo", onerror=remove_readonly)
            except:
                pass

if __name__ == "__main__":
    push_data_to_github()
