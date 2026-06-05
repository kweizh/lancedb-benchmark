import os
import sys
from solution import build_index, cross_lingual_search

# Mock ZEALT_RUN_ID for testing
os.environ["ZEALT_RUN_ID"] = "test_run"

def test():
    if not os.environ.get("COHERE_API_KEY"):
        print("Error: COHERE_API_KEY not set. Cannot run tests.")
        return

    print("Building index...")
    build_index()
    print("Index built.")

    # Test 1: English query
    print("\nTesting English query: 'Tell me about the Eiffel Tower'")
    results_en = cross_lingual_search("Tell me about the Eiffel Tower", k=3)
    for res in results_en:
        print(f"ID: {res['concept_id']}, Lang: {res['language']}, Text: {res['text'][:50]}...")
    
    # Check if multiple languages are present
    langs_en = set(res['language'] for res in results_en)
    print(f"Languages found: {langs_en}")
    
    # Test 2: Spanish query
    print("\nTesting Spanish query: 'Háblame de la Torre Eiffel'")
    results_es = cross_lingual_search("Háblame de la Torre Eiffel", k=3)
    for res in results_es:
        print(f"ID: {res['concept_id']}, Lang: {res['language']}, Text: {res['text'][:50]}...")
    
    langs_es = set(res['language'] for res in results_es)
    print(f"Languages found: {langs_es}")

if __name__ == "__main__":
    test()
