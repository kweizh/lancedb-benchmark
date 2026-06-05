from solution import exclude_brands, exclude_categories_and_price, not_like_search, complex_negation
import numpy as np

query = np.random.rand(16).tolist()
print("exclude_brands:")
print(exclude_brands(query, ["BrandA", "BrandB"], 2))

print("exclude_categories_and_price:")
print(exclude_categories_and_price(query, "Cat1", 500.0, 2))

print("not_like_search:")
print(not_like_search(query, "premium", 2))

print("complex_negation:")
print(complex_negation(query, 2))
