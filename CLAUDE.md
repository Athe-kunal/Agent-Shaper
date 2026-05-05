## Code Style

Follow the **Google coding style** throughout the codebase.

## Readability

- **Readability over cleverness** — prefer clear, simple code over complex or "clever" solutions
- **Same abstraction level** — keep code within a function at a consistent level of abstraction
- **No nested functions** — avoid defining functions inside other functions
- **No unnecessary one-liners** — don't extract single-line logic into a function unless it's reused meaningfully

## Structure & Modularity

- Break logic into **small, focused functions** with a low line count
- Group related functions into the **same file or folder** (module-driven development)
- For public data models, maintain them in a dedicated **`datamodels.py`** in the same folder

## Return Values

- Returning **one value** → return it directly
- Returning **two values** → a plain tuple is fine
- Returning **three or more values** → use a `NamedTuple` with proper type hints
  - If the `NamedTuple` is only used within the current file, make it private: `_MyResult`

## Logging

Always include the variable name when logging:

```python
log.info(f"{my_var=}")
```