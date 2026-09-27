def main() -> None:
    """Run the FastAPI app with uvicorn.

    Not wired up as a console script: pyproject.toml has no [project.scripts]
    entry pointing here. Call it directly if needed, e.g. `uv run python -c
    "from estimador_cag import main; main()"`; normal usage is
    `uvicorn app.main:app` instead.
    """
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
