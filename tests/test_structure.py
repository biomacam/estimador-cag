from pathlib import Path


def test_repository_structure():
    required_paths = [
        "app",
        "app/main.py",
        "app/config.py",
        "app/routers",
        "app/services",
        "app/schemas",
        "app/context",
        "tests",
        "pyproject.toml",
    ]

    missing = []

    for path in required_paths:
        if not Path(path).exists():
            missing.append(path)

    assert not missing, (
        f"Faltan elementos obligatorios de la estructura: {missing}"
    )