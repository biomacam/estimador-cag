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
- **Docker** y **Docker Compose** instalados
- Una **API key** de OpenAI o Anthropic
- Tener Python instalado


## Inicio rapido con Docker (recomendado)

1. Clonar el repositorio y entrar al directorio:
   ```bash
   cd estimator
   ```

2. Copiar el archivo de variables de entorno y configurar las API keys:
   ```bash
   cp .env.example .env
   # Editar .env y poner tu API key real
   ```

3. Construir y levantar el servicio:
   ```bash
   docker compose up --build
   ```

4. El servicio estara disponible en `http://localhost:8000`

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
Al final lo he solucionado añadiendo [tool.pytest.ini_options] al pyproject.toml con el parámetro pythonpath = ["."]. Ahora ya puedo ejecutar directamente "uv run pytest"
- **Error al hacer import structlog** 
En algunos ficheros obutve este error y tras pedirle a la IA que me lo solucionara me creó la carpeta src\estimador_cag con el fichero _init_.py que me solucionó el error.
- **Error en la pipeline**
Comento la siguiente linea en main.py porque la pipeline si no no pasa ya que no subo en el repo el fichero .env con el ApiKey de OpenAI. Después de consultar con la IA, parece un porblema de diseño porque el test health debe probar que fastapi funciona correctamente, no que la configuración de OPENAI sea correcta, o que el servicio de OPENAI esté funcionando correctamente.
    #    "environment": settings.APP_ENV,

## Otros problemas que me he encontrado, y he superado!

- ** Docker ** He tendio problemas al hacer lel build de la imagen de docker, peor con la ayuda d ela IA he podido resolver los paths de configuración que me faltaban en el dockerfile

- ** Python ** Apenas conozco python y eso me ha impedido un poco modificar el código por miedo a que dejar todo de funcionar, algunos cambios que intenté hacer  me rompían la compilación.

## Sensaciones ##
Aunque esto es un readme dejo este feedback sobre el ejercicio y en las siguientes ramas de próximas lecciones lo quitaré. Como primer ejercicio ha sido todfo un reto porque aparte de entender la arquitectura y el funcionamiento de la aplicación, el tema de la instalación del entorno de desarrollo con las dependencias me ha jugado alguna mala pasada (aún no sé porque he tenido que la carpeta src para evitar problemas con el import structlog). Soy nuevo en python, normalmente uso ADO, pero me he decidido por subir el repo a Github y además nunca había utilizado Docker porque en mi trabajo generalmente trabajamos con aplicaciones legacy de escritorio. Así que en cada paso me he ido encontrando con algún pequeño problema.
Lo bueno es que he aprendido mucho y sé que para el próximo voy a tener que estructurarme mejor los días de dedicación, teniendo en cuenta que además he de estudiar el contenido de la próxima lección. He visto bastante salto cualitativo ente el ejercicio de la sesión 1 y el ejercicio de la sesión 2.