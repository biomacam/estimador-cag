def main() -> None:
    """Entry point for the estimador-cag console script."""
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
