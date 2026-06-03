A common developer friction point in LanceDB is encountering `Unterminated string literal` errors when executing SQL filters that contain single quotes in the string payload.

You need to successfully execute a `table.update()` operation to change a `status` column to "resolved" where the `message` column exactly matches the string `I'm good`. 

**Constraints:**
- Must properly escape the single quote in the `where()` clause to prevent SQL syntax errors.
- Do not modify the underlying data format; the target row must actually contain the literal string `I'm good`.