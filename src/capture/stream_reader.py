# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
from collections import deque
from threading import Thread, Event
from queue import Queue, Full, Empty
from time import time
from typing import Optional
import asyncio
import os

import cv2
import numpy as np
from loguru import logger


class StreamReader:
    """RTSP/ONVIF stream capture with buffering and auto-reconnect."""

    def __init__(
        self,
        rtsp_url: str,
        target_fps: int = 10,
        buffer_size: int = 30,
        reconnect_interval: float = 5.0,
        debug: bool = False,
    ):
        self.rtsp_url = rtsp_url
        self.target_fps = target_fps
        self.buffer_size = buffer_size
        self.reconnect_interval = reconnect_interval
        self.debug = debug

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_queue: Queue = Queue(maxsize=buffer_size)
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._actual_fps: float = 0.0

    @property
    def fps(self) -> float:
        return self._actual_fps

    def start(self):
        logger.info(f"Starting stream reader for: {self.rtsp_url}")
        if self._cap is not None and self._cap.isOpened():
            logger.warning(f"Stream reader already running for: {self.rtsp_url}")
            return
        self._configure_ffmpeg()
        self._stop_event.clear()
        self._thread = Thread(target=self._read_loop, daemon=True, name="stream-reader")
        self._thread.start()
        logger.info(f"Stream reader thread started for: {self.rtsp_url}")

    async def start_async(self):
        """Async wrapper for start() to not block event loop."""
        if self._cap is not None and self._cap.isOpened():
            logger.warning(f"Stream reader already running for: {self.rtsp_url}")
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.start)
        logger.info(f"Stream reader async started for: {self.rtsp_url}")

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Stream reader stopped")

    def _configure_ffmpeg(self):
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|stimeout;5000000|timeout;5000000|max_delay;500000",
        )
        # Some cameras expose auxiliary streams before video; too low value stops before video packets.
        os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "1024")

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        except Exception:
            pass
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _read_single_frame(self) -> Optional[np.ndarray]:
        cap = self._open_capture()
        if cap is None:
            return None
        try:
            ret, frame = cap.read()
            if ret and frame is not None:
                return frame
            return None
        finally:
            cap.release()

    def _put_frame(self, frame: np.ndarray):
        try:
            self._frame_queue.put_nowait(frame)
        except Full:
            try:
                self._frame_queue.get_nowait()
            except Empty:
                pass
            try:
                self._frame_queue.put_nowait(frame)
            except Full:
                pass

    def read(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        try:
            frame = self._frame_queue.get(timeout=timeout)
            return frame
        except Empty:
            return None

    def _read_loop(self):
        interval = 1.0 / max(self.target_fps, 1)
        last_ts = time()
        fps_window = deque(maxlen=30)
        frame_count = 0

        logger.info(f"_read_loop started for {self.rtsp_url}, interval={interval:.3f}s")
        try:
            while not self._stop_event.is_set():
                now = time()
                elapsed = now - last_ts
                if elapsed < interval:
                    self._stop_event.wait(interval - elapsed)
                    continue

                retry_attempts = 0
                while retry_attempts < 3:
                    if self._cap is None or not self._cap.isOpened():
                        self._cap = self._open_capture()
                        if self._cap is not None:
                            logger.info(f"RTSP capture opened for {self.rtsp_url}")

                    if self._cap is not None and self._cap.isOpened():
                        ret, frame = self._cap.read()
                    else:
                        ret, frame = False, None

                    if ret and frame is not None:
                        if frame_count == 0:
                            logger.info(f"Connected to camera: {self.rtsp_url}")
                        frame_count += 1
                        now_read = time()
                        fps_window.append(now_read)
                        last_ts = now_read
                        self._put_frame(frame)
                        if len(fps_window) > 1:
                            elapsed_window = max(fps_window[-1] - fps_window[0], interval)
                            self._actual_fps = min(self.target_fps, (len(fps_window) - 1) / elapsed_window)
                        else:
                            self._actual_fps = 0.0
                        if frame_count % 1000 == 0:
                            logger.info(
                                f"_read_loop: {frame_count} frames, "
                                f"fps={self._actual_fps:.1f}"
                            )
                        break

                    retry_attempts += 1
                    logger.warning(f"Failed to read frame, retry {retry_attempts}/3")
                    if self._cap:
                        self._cap.release()
                        self._cap = None
                    self._stop_event.wait(0.5)

                if retry_attempts >= 3:
                    logger.error(f"Connection lost, reconnecting in {self.reconnect_interval}s...")
                    self._stop_event.wait(self.reconnect_interval)
        except Exception:
            logger.exception(f"_read_loop crashed for {self.rtsp_url}")
