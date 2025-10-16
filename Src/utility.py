import uuid
import os
from typing import Tuple

def paper_id_from_path(file_path: str) -> Tuple[str, str]:
    """Generate a deterministic UUID based on the file path."""
    # Define a namespace for UUID generation (can be any valid UUID)
    namespace = uuid.NAMESPACE_DNS
    # Use UUID version 5 for a deterministic, namespace-based ID
    return str(uuid.uuid5(namespace, os.path.abspath(file_path))), os.path.abspath(file_path)