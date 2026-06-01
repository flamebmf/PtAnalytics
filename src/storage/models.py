# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
import uuid
from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import String, Integer, Float, DateTime, Text, ForeignKey, JSON, Boolean, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    rtsp_url: Mapped[str] = mapped_column(Text, nullable=False)
    fps: Mapped[int] = mapped_column(Integer, default=10)
    enabled: Mapped[bool] = mapped_column(default=True)
    config_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    objects: Mapped[list["TrackedObject"]] = relationship(back_populates="camera")


class TrackedObject(Base):
    __tablename__ = "tracked_objects"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    camera_id: Mapped[str] = mapped_column(String(64), ForeignKey("cameras.id"), nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_name: Mapped[str] = mapped_column(String(32), nullable=False)
    first_seen: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_seen: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    plate_number: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    face_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    face_hash: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(512), nullable=True)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSON, nullable=True)
    name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    ignored: Mapped[bool] = mapped_column(default=False)
    appearance_count: Mapped[int] = mapped_column(Integer, default=1)

    camera: Mapped["Camera"] = relationship(back_populates="objects")
    frames: Mapped[list["FrameCapture"]] = relationship(back_populates="object", cascade="all, delete-orphan")


class FrameCapture(Base):
    __tablename__ = "frame_captures"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    object_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tracked_objects.id"), nullable=False)
    image_path: Mapped[str] = mapped_column(Text, nullable=False)
    bbox_x1: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y1: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_x2: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y2: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    object: Mapped["TrackedObject"] = relationship(back_populates="frames")


class CropSample(Base):
    __tablename__ = "crop_samples"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False)
    class_name: Mapped[str] = mapped_column(String(32), nullable=False)
    bbox_x1: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y1: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_x2: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y2: Mapped[int] = mapped_column(Integer, nullable=False)
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    phase: Mapped[str] = mapped_column(String(16), default="entry")
    is_val: Mapped[bool] = mapped_column(Boolean, default=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Event(Base):
    __tablename__ = "events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    object_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tracked_objects.id"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    trigger_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    action_result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
