import subprocess
import os

# Define the output text file
output_file = "ClasicalKmeans_GA_30points_10runs.txt"

# Define the script to run
script_name = "kmeans_GA_5.py"

# Dynamically get the absolute path to the script
current_directory = os.path.dirname(os.path.abspath(__file__))
script_path = os.path.join(current_directory, script_name)

# Verify that the script exists
if not os.path.exists(script_path):
    print(f"Error: Script '{script_path}' not found.")
    exit()

# Define the number of times to run the script
num_runs = 10

# Open the output file in write mode
with open(output_file, "w") as file:
    for i in range(1, num_runs + 1):
        print(f"Running {script_name} - Run {i}...")
        file.write(f"--- Run {i} ---\n\n")  # Header for each run

        # Run the script and capture stdout and stderr
        result = subprocess.run(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Write the results to the text file
        file.write(result.stdout)  # Script's standard output
        if result.stderr:
            file.write("\nErrors:\n")
            file.write(result.stderr)  # Include errors, if any
        
        file.write("\n\n")  # Add spacing between runs

print(f"\nAll {num_runs} runs completed. Results saved to '{output_file}'.")
