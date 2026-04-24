"""Script di verifica che il database sia accessibile."""
import os
print(f"[config] RENDER={os.environ.get('RENDER', 'not set')}")
print(f"[config] DB will be at: {'tmp/cinerec.db' if os.environ.get('RENDER') else 'local/database.db'}")
