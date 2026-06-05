import json
import sys
from solution import sweep

def main():
    try:
        # Call the sweep function
        results = sweep()
        
        # Convert integer keys to string keys as required (e.g., {"4": 0.78, "8": 0.91, "16": 0.97})
        formatted_results = {str(k): float(v) for k, v in results.items()}
        
        # Write the result to result.json
        output_path = '/home/user/myproject/result.json'
        with open(output_path, 'w') as f:
            json.dump(formatted_results, f, indent=4)
            
        print(f"Sweep completed successfully. Results written to {output_path}: {formatted_results}")
    except Exception as e:
        print(f"Error during sweep run: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
