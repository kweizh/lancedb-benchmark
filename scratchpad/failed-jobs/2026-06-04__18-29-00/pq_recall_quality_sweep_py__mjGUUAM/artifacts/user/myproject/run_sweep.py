"""CLI entrypoint: run the recall sweep and write result.json."""

import json
from solution import sweep


def main():
    result = sweep()
    # JSON requires string keys
    result_json = {str(k): v for k, v in result.items()}
    with open("/home/user/myproject/result.json", "w") as f:
        json.dump(result_json, f)
    print(result_json)


if __name__ == "__main__":
    main()