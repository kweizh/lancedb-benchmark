import os
import hashlib
import numpy as np
import lancedb
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash

app = Flask(__name__)
# Set a stable secret key for flash messages
app.secret_key = "lancedb-admin-dashboard-secret-key-12345"

# Open LanceDB connection
DB_PATH = "/home/user/myproject/data/lancedb"
db = lancedb.connect(DB_PATH)
table = db.open_table("products")

def escape_sql_string(val):
    """Safely escape single quotes for SQL filter strings."""
    return val.replace("'", "''")

def generate_deterministic_vector(product_id):
    """Generate a deterministic 32-d float32 vector based on product_id."""
    hasher = hashlib.sha256(product_id.encode('utf-8'))
    # Convert first 4 bytes of hash to an integer
    seed = int.from_bytes(hasher.digest()[:4], 'little')
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(32, dtype=np.float32).tolist()
    return vector

@app.route("/")
def index():
    try:
        # Load all products using to_pandas()
        df = table.to_pandas()
        if df.empty:
            products = []
        else:
            products = df.to_dict(orient='records')
    except Exception as e:
        flash(f"Error loading products: {e}")
        products = []
    return render_template("index.html", products=products)

@app.route("/product/new", methods=["GET"])
def new_product():
    return render_template("new.html")

@app.route("/product", methods=["POST"])
def create_product():
    product_id = request.form.get("id", "").strip()
    name = request.form.get("name", "").strip()
    category = request.form.get("category", "").strip()
    price_str = request.form.get("price", "0").strip()
    
    if not product_id or not name or not category or not price_str:
        flash("All fields are required.")
        return redirect(url_for("new_product"))
    
    try:
        price = float(price_str)
    except ValueError:
        flash("Price must be a valid number.")
        return redirect(url_for("new_product"))
    
    # Check if id already exists safely using pandas
    try:
        df = table.to_pandas()
        if not df.empty and product_id in df["id"].values:
            flash(f"Product with ID '{product_id}' already exists.")
            return redirect(url_for("new_product"))
    except Exception as e:
        flash(f"Error checking existing products: {e}")
        return redirect(url_for("new_product"))
    
    # Generate deterministic 32-dim vector
    vector = generate_deterministic_vector(product_id)
    
    try:
        table.add([{
            "id": product_id,
            "name": name,
            "category": category,
            "price": price,
            "vector": vector
        }])
        flash(f"Product '{name}' created successfully.")
    except Exception as e:
        flash(f"Error creating product: {e}")
        return redirect(url_for("new_product"))
        
    return redirect(url_for("index"))

@app.route("/product/<path:product_id>/edit", methods=["GET"])
def edit_product(product_id):
    escaped = escape_sql_string(product_id)
    try:
        rows = table.search().where(f"id = '{escaped}'").to_list()
        if not rows:
            flash(f"Product with ID '{product_id}' not found.")
            return redirect(url_for("index"))
        return render_template("edit.html", product=rows[0])
    except Exception as e:
        flash(f"Error fetching product: {e}")
        return redirect(url_for("index"))

@app.route("/product/<path:product_id>", methods=["POST"])
def update_product(product_id):
    name = request.form.get("name", "").strip()
    category = request.form.get("category", "").strip()
    price_str = request.form.get("price", "0").strip()
    
    if not name or not category or not price_str:
        flash("All fields are required.")
        return redirect(url_for("edit_product", product_id=product_id))
    
    try:
        price = float(price_str)
    except ValueError:
        flash("Price must be a valid number.")
        return redirect(url_for("edit_product", product_id=product_id))
    
    escaped = escape_sql_string(product_id)
    try:
        # Verify product exists
        rows = table.search().where(f"id = '{escaped}'").to_list()
        if not rows:
            flash(f"Product with ID '{product_id}' not found.")
            return redirect(url_for("index"))
        
        # Perform update
        table.update(
            where=f"id = '{escaped}'",
            values={
                "name": name,
                "category": category,
                "price": price
            }
        )
        flash(f"Product '{product_id}' updated successfully.")
    except Exception as e:
        flash(f"Error updating product: {e}")
        return redirect(url_for("edit_product", product_id=product_id))
        
    return redirect(url_for("index"))

@app.route("/product/<path:product_id>/delete", methods=["POST"])
def delete_product(product_id):
    escaped = escape_sql_string(product_id)
    try:
        # Verify product exists
        rows = table.search().where(f"id = '{escaped}'").to_list()
        if not rows:
            flash(f"Product with ID '{product_id}' not found.")
            return redirect(url_for("index"))
            
        table.delete(f"id = '{escaped}'")
        flash(f"Product '{product_id}' deleted successfully.")
    except Exception as e:
        flash(f"Error deleting product: {e}")
    return redirect(url_for("index"))

@app.route("/api/products", methods=["GET"])
def api_products():
    try:
        df = table.to_pandas()
        if df.empty:
            return jsonify([])
        
        # Keep only the columns we need
        products_df = df[["id", "name", "category", "price"]]
        products = products_df.to_dict(orient="records")
        return jsonify(products)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
