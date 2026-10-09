from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import os

API_KEY_NAME = "X-EdgeMind-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    # In production, this should be a secure secret manager call
    expected_api_key = os.environ.get("EDGEMIND_API_KEY", "edgemind_secure_key_2026")
    if api_key_header == expected_api_key:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )
