LanceDB provides an Embedding API registry that automatically manages embedding generation behind the scenes when properly defined in a schema.

You need to define a Pydantic schema using `LanceModel` and the `get_registry` embedding function for OpenAI's `text-embedding-3-small`, initialize a LanceDB table, and insert raw text records so the vectors are generated automatically. 

**Constraints:**
- Must use `lancedb.pydantic.LanceModel` for the schema definition.
- Must define the text column as a `SourceField()` and the vector column as a `VectorField()` using the registry function.
- Do not manually generate the vector embeddings prior to calling `table.add()`.