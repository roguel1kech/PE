import os
from app.observability.langfuse_client import langfuse, PUBLIC_KEY, SECRET_KEY, BASE_URL

print("PUBLIC_KEY:", PUBLIC_KEY)
print("SECRET_KEY:", SECRET_KEY[:6] + "..." if SECRET_KEY else None)
print("BASE_URL:", BASE_URL)
