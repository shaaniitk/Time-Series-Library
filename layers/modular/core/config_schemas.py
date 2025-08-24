"""Compatibility shim: re-export schemas from configs.schemas for modular imports.

This file exists to preserve import paths used by tests and scripts:
`from layers.modular.core.config_schemas import ...`.
"""
from configs.schemas import *  # noqa: F401,F403
