import json
from solution import sweep

def main():
    results = sweep()
    json_results = {str(k): v for k, v in results.items()}
    with open('/home/user/myproject/result.json', 'w') as f:
        json.dump(json_results, f)

if __name__ == "__main__":
    main()
