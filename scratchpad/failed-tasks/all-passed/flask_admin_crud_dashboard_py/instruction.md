# Flask CRUD Admin Dashboard for LanceDB Products

## Background
You are building a small browser-based admin dashboard over a LanceDB `products` table. The container ships with a pre-seeded LanceDB store at `/home/user/myproject/data/lancedb` containing a `products` table with 5 deterministic seed rows. Each row has `id` (string), `name` (string), `category` (string), `price` (float64), and `vector` (32-dim float vector). Your job is to expose a Flask web application that lets a human operator list, create, edit, and delete product rows through HTML pages, plus a small JSON dump endpoint that downstream tooling can poll.

## Requirements
- Build a Flask application that opens the existing LanceDB table at `/home/user/myproject/data/lancedb/products` (do **NOT** drop or overwrite it; new rows must extend the seeded set).
- Provide an HTML index page that lists all current products in a `<table>` element, including the columns `id`, `name`, `category`, and `price`, and per-row links/buttons to edit or delete each product.
- Provide an HTML form to create a new product. Submitting the form must insert a new row into LanceDB.
- Provide an HTML form to edit an existing product. Submitting the form must update the matching row in LanceDB.
- Provide a delete action that removes the matching row from LanceDB.
- Provide a JSON endpoint that returns the full current state of the table as an array of `{id, name, category, price}` objects (the `vector` column does not need to be exposed).
- Every new row inserted from the create form must include a deterministic 32-dim numeric vector so the LanceDB schema stays valid. The dashboard itself does NOT need to expose any vector search; this task is pure CRUD.

## Implementation Hints
- Open the table with `lancedb.connect("/home/user/myproject/data/lancedb").open_table("products")`. The schema is fixed (`id`, `name`, `category`, `price`, `vector` of fixed-size 32-d float32).
- Use `table.add([row_dict])` to insert. Use `table.update(where="id = '...'", values={...})` to update individual fields. Use `table.delete("id = '...'")` to delete.
- Prefer `table.update(where=..., values={...})` over `values_sql=` to avoid the apostrophe-escaping footgun documented in lancedb issue #1429.
- Use `table.to_pandas()` (or `table.search().limit(N).to_list()` with a large enough N) to read the full table when rendering the index page or producing the JSON dump.
- For the create form, generate a 32-d vector from `numpy.random.default_rng(<some seed derived from the new id>)` so the insert is reproducible if the same id is reused.
- Jinja templates, vanilla HTML forms, and `application/x-www-form-urlencoded` posts are all fine; the verifier only checks the rendered HTML structure and the JSON endpoint, not your template engine choice.

## Acceptance Criteria
- Project path: /home/user/myproject
- Start command: python3 app.py
- Port: 5000
- API endpoints / routes:
  - `GET /` — Returns HTML (status 200). The response body MUST contain a `<table>` element whose rows expose, at minimum, each product's `id`, `name`, `category`, and `price`. Each existing-product row MUST also contain controls (links or buttons) that lead to the edit and delete actions for that product.
  - `GET /product/new` — Returns HTML (status 200) with a form for creating a new product (fields: `id`, `name`, `category`, `price`).
  - `POST /product` — Accepts the create form submission and inserts a new row into the LanceDB `products` table. May return HTML or a redirect (any 2xx or 3xx status). The newly inserted product MUST appear in subsequent `GET /` and `GET /api/products` responses.
  - `GET /product/<id>/edit` — Returns HTML (status 200) for editing the product whose primary key is `<id>`. The form must be pre-filled with the current `name`, `category`, and `price`.
  - `POST /product/<id>` — Accepts the edit form submission and updates the matching row in LanceDB. The updated values MUST appear in subsequent `GET /` and `GET /api/products` responses.
  - `POST /product/<id>/delete` — Deletes the matching row from LanceDB. The deleted product MUST disappear from subsequent `GET /` and `GET /api/products` responses.
  - `GET /api/products` — Returns `application/json` (status 200) with a JSON array of objects, each containing at least the keys `id`, `name`, `category`, `price`. The order does not matter; the verifier matches by `id`.
- Persistence: All create/update/delete operations MUST be reflected in the actual LanceDB table on disk (the verifier reopens the table and inspects it).
- Do NOT drop, recreate, or overwrite the existing `products` table; new rows must extend the build-time seed.

