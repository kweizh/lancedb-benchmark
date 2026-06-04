import os
import re
import unittest
import lancedb
from solution import search_with_snippets

class TestSearchWithSnippets(unittest.TestCase):
    def setUp(self):
        # Ensure LANCE_TABLE is set
        os.environ["LANCE_TABLE"] = "articles"
        self.db_path = "/home/user/myproject/data"
        self.table_name = "articles"
        self.db = lancedb.connect(self.db_path)
        self.tbl = self.db.open_table(self.table_name)

    def test_schema_and_types(self):
        results = search_with_snippets("indigo", 3, 120)
        self.assertLessEqual(len(results), 3)
        for row in results:
            self.assertIn("id", row)
            self.assertIn("score", row)
            self.assertIn("snippet", row)
            self.assertIn("snippet_offset", row)
            
            self.assertIsInstance(row["id"], int)
            self.assertIsInstance(row["score"], float)
            self.assertIsInstance(row["snippet"], str)
            self.assertIsInstance(row["snippet_offset"], int)

    def test_sorting_order(self):
        results = search_with_snippets("indigo", 5, 120)
        scores = [r["score"] for r in results]
        # Check that scores are in descending order
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_unhighlighted_length_and_content(self):
        # Test that unhighlighted snippet is exactly the substring of body
        results = search_with_snippets("indigo", 5, 120)
        for row in results:
            body = self.tbl.search().where(f"id = {row['id']}").to_list()[0]["body"]
            snippet = row["snippet"]
            offset = row["snippet_offset"]
            
            # Remove <mark> and </mark>
            unhighlighted = snippet.replace("<mark>", "").replace("</mark>", "")
            
            # Check length constraint
            self.assertLessEqual(len(unhighlighted), 120)
            
            # Check that it matches the body slice exactly
            expected_slice = body[offset : offset + len(unhighlighted)]
            self.assertEqual(unhighlighted, expected_slice)

    def test_exact_case_preserved_highlighting(self):
        # Search for a term that might have different casings or just check case preservation
        results = search_with_snippets("indigo", 5, 120)
        for row in results:
            snippet = row["snippet"]
            # Extract all marked contents
            marks = re.findall(r"<mark>(.*?)</mark>", snippet)
            for mark in marks:
                # Should be case-insensitive equal to "indigo"
                self.assertEqual(mark.lower(), "indigo")
                
                # Check that it matches the original case in the body at that offset
                body = self.tbl.search().where(f"id = {row['id']}").to_list()[0]["body"]
                offset = row["snippet_offset"]
                # Find where the mark is in the unhighlighted snippet
                unhighlighted_snippet = snippet.replace("<mark>", "").replace("</mark>", "")
                mark_idx_in_unhighlighted = unhighlighted_snippet.find(mark)
                self.assertNotEqual(mark_idx_in_unhighlighted, -1)
                
                original_char_idx = offset + mark_idx_in_unhighlighted
                expected_original = body[original_char_idx : original_char_idx + len(mark)]
                self.assertEqual(mark, expected_original)

    def test_token_fallback(self):
        # Search for a phrase where the exact phrase doesn't exist but tokens do
        # "indigo nonexistentword" -> exact phrase doesn't exist, but "indigo" does.
        results = search_with_snippets("indigo nonexistentword", 5, 120)
        self.assertTrue(len(results) > 0)
        for row in results:
            snippet = row["snippet"]
            self.assertIn("<mark>", snippet)
            # The highlighted term should be "indigo" (case-insensitive)
            marks = re.findall(r"<mark>(.*?)</mark>", snippet)
            for mark in marks:
                self.assertEqual(mark.lower(), "indigo")

    def test_complete_fallback(self):
        # Search for "wanders", which matches documents containing "wandered" via stemming,
        # but does not exist in the documents.
        results = search_with_snippets("wanders", 3, 120)
        self.assertTrue(len(results) > 0)
        for row in results:
            snippet = row["snippet"]
            self.assertNotIn("<mark>", snippet)
            self.assertEqual(row["snippet_offset"], 0)
            self.assertLessEqual(len(snippet), 120)

    def test_stemming_token_fallback(self):
        # Search for "wanders wandered", which should highlight "wandered"
        results = search_with_snippets("wanders wandered", 3, 120)
        self.assertTrue(len(results) > 0)
        for row in results:
            snippet = row["snippet"]
            self.assertIn("<mark>", snippet)
            marks = re.findall(r"<mark>(.*?)</mark>", snippet)
            for mark in marks:
                self.assertEqual(mark.lower(), "wandered")

if __name__ == "__main__":
    unittest.main()
