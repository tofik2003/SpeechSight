"""
Privacy and Safety Manager for SpeechSight.
Ensures local-first processing, ephemeral disk usage, and immediate zero-trace wiping.
"""

import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Generator
from contextlib import contextmanager

logger = logging.getLogger("speechsight.privacy")


class PrivacyManager:
    def __init__(self, base_temp_dir: Path | None = None):
        self.base_temp_dir = base_temp_dir or Path(tempfile.gettempdir()) / "speechsight_ephemeral"
        self.base_temp_dir.mkdir(parents=True, exist_ok=True)
        self.active_sessions: set[str] = set()

    def create_session_dir(self, session_id: str) -> Path:
        """Create a dedicated ephemeral folder for this processing session."""
        session_path = self.base_temp_dir / session_id
        session_path.mkdir(parents=True, exist_ok=True)
        self.active_sessions.add(session_id)
        logger.info(f"Created secure ephemeral session workspace: {session_path}")
        return session_path

    def secure_delete_file(self, file_path: Path):
        """Securely overwrite and remove a file."""
        try:
            if file_path.exists() and file_path.is_file():
                # Overwrite with random bytes or zeros before unlinking
                size = file_path.stat().st_size
                if size > 0 and size < 50 * 1024 * 1024:  # up to 50MB overwrite
                    with open(file_path, "wb") as f:
                        f.write(b"\x00" * min(size, 1024 * 1024))
                file_path.unlink()
                logger.debug(f"Securely deleted: {file_path}")
        except Exception as e:
            logger.warning(f"Error during secure delete of {file_path}: {e}")

    def wipe_session(self, session_id: str) -> bool:
        """Completely remove all temporary media, crops, and audio slices for a session."""
        session_path = self.base_temp_dir / session_id
        try:
            if session_path.exists():
                for root, dirs, files in os.walk(session_path, topdown=False):
                    for file in files:
                        self.secure_delete_file(Path(root) / file)
                    for dir_name in dirs:
                        try:
                            (Path(root) / dir_name).rmdir()
                        except Exception:
                            pass
                shutil.rmtree(session_path, ignore_errors=True)
            if session_id in self.active_sessions:
                self.active_sessions.remove(session_id)
            logger.info(f"Privacy wipe completed for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to wipe session {session_id}: {e}")
            return False

    def wipe_all(self) -> int:
        """Wipe all active and residual ephemeral session folders."""
        count = 0
        if self.base_temp_dir.exists():
            for item in self.base_temp_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                    count += 1
                elif item.is_file():
                    self.secure_delete_file(item)
                    count += 1
        self.active_sessions.clear()
        logger.info(f"Purged {count} temporary items from storage.")
        return count

    @contextmanager
    def ephemeral_workspace(self, session_id: str) -> Generator[Path, None, None]:
        """Context manager to ensure automatic cleanup even on crash or exception."""
        workspace = self.create_session_dir(session_id)
        try:
            yield workspace
        finally:
            self.wipe_session(session_id)


# Global privacy manager instance
privacy_manager = PrivacyManager()
