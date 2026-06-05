import os
import pytest
from solution import autocomplete

def test_autocomplete_more_than_k():
    # 'crystal' has 17 SQL matches. With k=10, we should get exactly 10 'prefix' results.
    results = autocomplete("crystal", k=10)
    assert len(results) == 10
    
    # Check that all are "prefix"
    for item in results:
        assert item["source"] == "prefix"
        assert item["title"].lower().startswith("crystal")
        assert isinstance(item["id"], int)
        assert isinstance(item["title"], str)
        assert isinstance(item["popularity"], float)
        assert isinstance(item["source"], str)
        
    # Check that they are sorted by popularity descending
    popularities = [item["popularity"] for item in results]
    assert popularities == sorted(popularities, reverse=True)

def test_autocomplete_fewer_than_k():
    # 'crystal' has 17 SQL matches. With k=20, we should get 17 'prefix' and 3 'semantic' results.
    results = autocomplete("crystal", k=20)
    assert len(results) == 20
    
    prefix_items = [item for item in results if item["source"] == "prefix"]
    semantic_items = [item for item in results if item["source"] == "semantic"]
    
    assert len(prefix_items) == 17
    assert len(semantic_items) == 3
    
    # Check sorting of prefix items
    prefix_pops = [item["popularity"] for item in prefix_items]
    assert prefix_pops == sorted(prefix_pops, reverse=True)
    
    # Check that semantic items exclude prefix item IDs
    prefix_ids = {item["id"] for item in prefix_items}
    for item in semantic_items:
        assert item["id"] not in prefix_ids
        assert isinstance(item["id"], int)
        assert isinstance(item["title"], str)
        assert isinstance(item["popularity"], float)
        assert isinstance(item["source"], str)

def test_autocomplete_zero_matches():
    # 'meridian' has 0 SQL matches. With k=10, we should get exactly 10 'semantic' results.
    results = autocomplete("meridian", k=10)
    assert len(results) == 10
    
    for item in results:
        assert item["source"] == "semantic"
        assert isinstance(item["id"], int)
        assert isinstance(item["title"], str)
        assert isinstance(item["popularity"], float)
        assert isinstance(item["source"], str)

def test_autocomplete_case_insensitive():
    results1 = autocomplete("CrYsTaL", k=10)
    results2 = autocomplete("crystal", k=10)
    
    assert len(results1) == len(results2)
    for r1, r2 in zip(results1, results2):
        assert r1["id"] == r2["id"]
        assert r1["title"] == r2["title"]
        assert r1["popularity"] == r2["popularity"]
        assert r1["source"] == r2["source"]
