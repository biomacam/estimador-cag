# Estimator CAG - Servicio de Estimacion de Software con IA

Servicio de estimacion de proyectos de software impulsado por IA, utilizando una arquitectura **Cache Augmented Generation (CAG)**.

## Que es CAG y por que lo usamos

CAG (Cache Augmented Generation) es un patron de arquitectura donde el contexto relevante se inyecta directamente en el prompt del LLM como texto estatico. En esta fase del proyecto, las estimaciones de referencia se incluyen como ejemplos dentro del prompt del sistema, sin necesidad de una base de datos vectorial ni busqueda semantica.

Este enfoque es ideal para empezar porque:
- Es simple de implementar y depurar
- No requiere infraestructura adicional (ni embeddings, ni vector stores)
- Funciona bien cuando el volumen de contexto es manejable (pocos ejemplos)

En modulos posteriores del master, este servicio evolucionara a una arquitectura **RAG** (Retrieval Augmented Generation) con base de datos vectorial para manejar un volumen mayor de ejemplos.

## Requisitos previos

- Una **API key** de OpenAI o Anthropic
- Tener Python instalado

## Alternativa: ejecucion local sin Docker

```bash
uv sync
# Configurar .env con tus API keys
uv run uvicorn app.main:app --reload
```

## Estructura del proyecto

```
estimador-cag/
├── app/
│   ├── main.py            # Aplicacion FastAPI, health check, CORS
│   ├── config.py           # Configuracion con Pydantic Settings
│   ├── routers/
│   │   └── estimations.py  # Endpoint POST /api/v1/estimate
│   ├── services/
│   │   └── llm_service.py  # Logica de negocio, llamadas al LLM
│   ├── schemas/
│   │   └── estimation.py   # Modelos Pydantic (request/response)
│   └── context/
│       └── examples.py     # Ejemplos de estimacion (contexto CAG)
├── tests/
│   └── test_health.py      # Tests basicos
└── pyproject.toml          # Dependencias y configuracion
└── src/estimador_cag       
   └── _init_.py            # este fichero y la carpeta padre me la ha creado la IA para   #resolver un problema que había con los import structlog en varios ficheros
```
## Documentacion interactiva

Con el servicio corriendo, accede a la documentacion Swagger UI en:

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---
## Problemas encontrados con el repo

- **Error en  tests**
Al ejecutar los tests con uv run pytest desde la consola me daba el error de que en "from app.main import app" app no existía. Tuve que ejectar uv run "python -m pytest" para que leyera la estructura de directorios desde el directorio actual.
Al final lo he solucionado añadiendo [tool.pytest.ini_options] al pyproject.toml con el parámetro pythonpath = ["."]. Ahora ya puedo ejecutar directamente "uv python test"
- **Error al hacer import structlog** 
En algunos ficheros obutve este error y tras pedirle a la IA que me lo solucionara me creó la carpeta src\estimador_cag con el fichero _init_.py que me solucionó el error.
- **Error en la pipeline**
Comento la siguiente linea en main.py porque la pipeline si no no pasa ya que no subo en el repo el fichero .env con el ApiKey de OpenAI. Después de consultar con la IA, parece un porblema de diseño porque el test health debe probar que fastapi funciona correctamente, no que la configuración de OPENAI sea correcta, o que el servicio de OPENAI esté funcionando correctamente.
    #    "environment": settings.APP_ENV,
