import os
import hashlib
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, redirect, url_for, jsonify
import lancedb

app = Flask(__name__)
DB_PATH = "/home/user/myproject/data/lancedb"

def get_table():
    db = lancedb.connect(DB_PATH)
    return db.open_table("products")

def generate_vector(id_str):
    # Deterministic vector based on id
    seed = int(hashlib.md5(id_str.encode('utf-8')).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.random(32).astype(np.float32).tolist()

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Products</title></head>
<body>
    <h1>Products</h1>
    <a href="/product/new">Create New Product</a>
    <table border="1">
        <thead>
            <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Category</th>
                <th>Price</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody>
            {% for p in products %}
            <tr>
                <td>{{ p.id }}</td>
                <td>{{ p.name }}</td>
                <td>{{ p.category }}</td>
                <td>{{ p.price }}</td>
                <td>
                    <a href="/product/{{ p.id }}/edit">Edit</a>
                    <form action="/product/{{ p.id }}/delete" method="POST" style="display:inline;">
                        <button type="submit">Delete</button>
                    </form>
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>
"""

NEW_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>New Product</title></head>
<body>
    <h1>New Product</h1>
    <form action="/product" method="POST">
        <label>ID:</label> <input type="text" name="id" required /><br/>
        <label>Name:</label> <input type="text" name="name" required /><br/>
        <label>Category:</label> <input type="text" name="category" required /><br/>
        <label>Price:</label> <input type="number" step="0.01" name="price" required /><br/>
        <button type="submit">Create</button>
    </form>
    <a href="/">Cancel</a>
</body>
</html>
"""

EDIT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Edit Product</title></head>
<body>
    <h1>Edit Product</h1>
    <form action="/product/{{ p.id }}" method="POST">
        <label>ID:</label> <input type="text" name="id" value="{{ p.id }}" readonly /><br/>
        <label>Name:</label> <input type="text" name="name" value="{{ p.name }}" required /><br/>
        <label>Category:</label> <input type="text" name="category" value="{{ p.category }}" required /><br/>
        <label>Price:</label> <input type="number" step="0.01" name="price" value="{{ p.price }}" required /><br/>
        <button type="submit">Update</button>
    </form>
    <a href="/">Cancel</a>
</body>
</html>
"""

@app.route("/")
def index():
    table = get_table()
    df = table.to_pandas()
    products = df.to_dict(orient="records")
    return render_template_string(INDEX_TEMPLATE, products=products)

@app.route("/product/new")
def new_product():
    return render_template_string(NEW_TEMPLATE)

@app.route("/product", methods=["POST"])
def create_product():
    id_str = request.form["id"]
    name = request.form["name"]
    category = request.form["category"]
    price = float(request.form["price"])
    
    vec = generate_vector(id_str)
    
    table = get_table()
    table.add([{
        "id": id_str,
        "name": name,
        "category": category,
        "price": price,
        "vector": vec
    }])
    
    return redirect("/")

@app.route("/product/<id_str>/edit")
def edit_product(id_str):
    table = get_table()
    df = table.to_pandas()
    product = df[df["id"] == id_str].to_dict(orient="records")
    if not product:
        return "Not found", 404
    return render_template_string(EDIT_TEMPLATE, p=product[0])

@app.route("/product/<id_str>", methods=["POST"])
def update_product(id_str):
    name = request.form["name"]
    category = request.form["category"]
    price = float(request.form["price"])
    
    table = get_table()
    # Update row
    # Using dictionary for values
    table.update(where=f"id = '{id_str}'", values={
        "name": name,
        "category": category,
        "price": price
    })
    
    return redirect("/")

@app.route("/product/<id_str>/delete", methods=["POST"])
def delete_product(id_str):
    table = get_table()
    table.delete(f"id = '{id_str}'")
    return redirect("/")

@app.route("/api/products")
def api_products():
    table = get_table()
    df = table.to_pandas()
    df = df[["id", "name", "category", "price"]]
    products = df.to_dict(orient="records")
    return jsonify(products)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
