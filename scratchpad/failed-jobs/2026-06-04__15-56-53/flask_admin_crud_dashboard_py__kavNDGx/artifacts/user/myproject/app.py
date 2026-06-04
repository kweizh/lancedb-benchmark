import os
import numpy as np
import lancedb
from flask import Flask, render_template, request, redirect, url_for, jsonify, abort

DB_PATH = "/home/user/myproject/data/lancedb"
TABLE_NAME = "products"
VECTOR_DIM = 32

app = Flask(__name__)


def get_table():
    db = lancedb.connect(DB_PATH)
    return db.open_table(TABLE_NAME)


def make_vector(seed_str: str) -> list:
    """Generate a deterministic 32-dim float32 vector from a string seed."""
    seed = abs(hash(seed_str)) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(VECTOR_DIM).astype("float32").tolist()


# ---------------------------------------------------------------------------
# Index – list all products
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    tbl = get_table()
    df = tbl.to_pandas()
    products = df[["id", "name", "category", "price"]].to_dict(orient="records")
    return render_template("index.html", products=products)


# ---------------------------------------------------------------------------
# Create – show form
# ---------------------------------------------------------------------------
@app.route("/product/new")
def product_new():
    return render_template("new_product.html")


# ---------------------------------------------------------------------------
# Create – handle form submission
# ---------------------------------------------------------------------------
@app.route("/product", methods=["POST"])
def product_create():
    prod_id = request.form.get("id", "").strip()
    name = request.form.get("name", "").strip()
    category = request.form.get("category", "").strip()
    price_str = request.form.get("price", "0").strip()

    if not prod_id or not name or not category:
        abort(400, "id, name, and category are required")

    try:
        price = float(price_str)
    except ValueError:
        abort(400, "price must be a number")

    vector = make_vector(prod_id)
    row = {
        "id": prod_id,
        "name": name,
        "category": category,
        "price": price,
        "vector": vector,
    }

    tbl = get_table()
    tbl.add([row])
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Edit – show pre-filled form
# ---------------------------------------------------------------------------
@app.route("/product/<prod_id>/edit")
def product_edit(prod_id):
    tbl = get_table()
    df = tbl.to_pandas()
    rows = df[df["id"] == prod_id]
    if rows.empty:
        abort(404, f"Product {prod_id!r} not found")
    product = rows.iloc[0][["id", "name", "category", "price"]].to_dict()
    return render_template("edit_product.html", product=product)


# ---------------------------------------------------------------------------
# Edit – handle form submission
# ---------------------------------------------------------------------------
@app.route("/product/<prod_id>", methods=["POST"])
def product_update(prod_id):
    name = request.form.get("name", "").strip()
    category = request.form.get("category", "").strip()
    price_str = request.form.get("price", "0").strip()

    if not name or not category:
        abort(400, "name and category are required")

    try:
        price = float(price_str)
    except ValueError:
        abort(400, "price must be a number")

    tbl = get_table()
    # Escape single quotes in values to avoid SQL injection issues
    safe_name = name.replace("'", "''")
    safe_category = category.replace("'", "''")
    tbl.update(
        where=f"id = '{prod_id}'",
        values={"name": name, "category": category, "price": price},
    )
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Delete – handle form submission
# ---------------------------------------------------------------------------
@app.route("/product/<prod_id>/delete", methods=["POST"])
def product_delete(prod_id):
    tbl = get_table()
    tbl.delete(f"id = '{prod_id}'")
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------
@app.route("/api/products")
def api_products():
    tbl = get_table()
    df = tbl.to_pandas()
    records = df[["id", "name", "category", "price"]].to_dict(orient="records")
    return jsonify(records)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
