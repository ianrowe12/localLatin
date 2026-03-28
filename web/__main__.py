"""Entry point: python -m web"""

import os
import uvicorn

from web.config import load_settings

config_path = os.environ.get("LOCALLATIN_CONFIG", None)
settings = load_settings(config_path)

uvicorn.run(
    "web.app:create_app",
    factory=True,
    host=settings.app.host,
    port=settings.app.port,
    reload=settings.app.debug,
)
