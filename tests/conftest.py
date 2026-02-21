"""
conftest.py
Configuração global do pytest: marcas personalizadas e fixtures compartilhadas.
"""

import pytest


def pytest_configure(config):
    """Registra marcas personalizadas para evitar warnings."""
    config.addinivalue_line(
        "markers",
        "integration: testes de integração que usam o dataset real ou serviços externos",
    )
