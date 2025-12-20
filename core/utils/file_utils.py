import os

def get_safe_filename(name):
    """Remove special chars and spaces to build a safe filename/folder name."""
    return "".join([c for c in name if c.isalpha() or c.isdigit() or c in (' ', '_', '-')]).rstrip().replace(' ', '_')
