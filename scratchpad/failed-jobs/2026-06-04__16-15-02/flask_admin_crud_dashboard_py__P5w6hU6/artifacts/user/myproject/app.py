import lancedb
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify

app = Flask(__name__)

DB_PATH = "/home/user/myproject/data/lancedb"
TABLE_NAME = "products"
VECTOR_DIM = 32


def get_table():
    db = lancedb.connect(DB_PATH)
    return db.open_table(TABLE_NAME)


def make_vector(product_id: str):
    """Generate a deterministic 32-dim vector from the product id."""
    seed = hash(product_id) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    return rng.standard_normal(VECTOR_DIM).astype(np.float32).tolist()


def get_all_products():
    tbl = get_table()
    df = tbl.to_pandas()
    return df.to_dict("records")


@app.route("/")
def index():
    products = get_all_products()
    return render_template("index.html", products=products)


@app.route("/product/new")
def new_product():
    return render_template("form.html", action=url_for("create_product"), product=None)


@app.route("/product", methods=["POST"])
def create_product():
    product_id = request.form["id"]
    name = request.form["name"]
    category = request.form["category"]
    price = float(request.form["price"])

    row = {
        "id": product_id,
        "name": name,
        "category": category,
        "price": price,
        "vector": make_vector(product_id),
    }
    tbl = get_table()
    tbl.add([row])
    return redirect(url_for("index"))


@app.route("/product/<pid>/edit")
def edit_product(pid):
    products = get_all_products()
    product = next((p for p in products if p["id"] == pid), None)
    if product is None:
        return "Product not found", 404
    return render_template("form.html", action=url_for("update_product", pid=pid), product=product)


@app.route("/product/<pid>", methods=["POST"])
def update_product(pid):
    name = request.form["name"]
    category = request.form["category"]
    price = float(request.form["price"])

    tbl = get_table()
    tbl.update(where=f"id = '{pid}'", values={"name": name, "category": category, "price": price})
    return redirect(url_for("index"))


@app.route("/product/<pid>/delete", methods=["POST"])
def delete_product(pid):
    tbl = get_table()
    tbl.delete(f"id = '{pid}'")
    return redirect(url_for("index"))


@app.route("/api/products")
def api_products():
    products = get_all_products()
    result = [
        {"id": p["id"], "name": p["name"], "category": p["category"], "price": p["price"]}
        for p in products
    ]
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)