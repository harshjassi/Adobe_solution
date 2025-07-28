import subprocess
import sys

def run_script(script_name):
    """Run a Python script"""
    print(f"Running {script_name}...")
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"Error running {script_name}")
        sys.exit(1)
    print(f"✓ {script_name} completed\n")

def main():
    print("🚀 Starting document analysis pipeline...\n")
    
    # Run the three scripts in order
    run_script("paragraph_extractor.py")
    run_script("merge_heading.py") 
    run_script("generate_output.py")
    
    print("🎉 All done! Check test_pdf/output.json for results")

if __name__ == "__main__":
    main()