LanceDB is built on the Arrow columnar format and enforces strict PyArrow typing. A common pitfall is passing standard Python `float64` lists to a `float32` vector column, resulting in hard-to-debug schema mismatches.

You need to define an explicit PyArrow schema featuring a 128-dimensional `float32` vector column, and successfully insert a list of standard Python floats without triggering an Arrow type error.

**Constraints:**
- The schema must be defined using `pyarrow` directly, not Pydantic.
- The vector column type must be specifically set to `pa.list_(pa.float32(), 128)`.
- You must correctly cast or format the incoming Python float data to `float32` during the `table.add()` operation.