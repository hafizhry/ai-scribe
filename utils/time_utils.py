from datetime import timedelta

def format_time(seconds):
    """Convert seconds to MM:SS format"""
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"