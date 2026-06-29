"""Tests for AttachmentManager."""

import os
import shutil
import tempfile

import pytest

from kesoku.gateway.attachment_manager import AttachmentManager
from kesoku.utils.async_fs import async_read_bytes


@pytest.fixture
def temp_sessions_dir():
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.mark.anyio
async def test_save_attachment_with_bytes(temp_sessions_dir):
    manager = AttachmentManager(sessions_dir=temp_sessions_dir)
    filename = "test_file.txt"
    workspace = "test_ws"
    data = b"hello world"

    result = await manager.save_attachment(filename=filename, workspace_name=workspace, data=data)

    assert result["filename"] == "test_file.txt"
    assert os.path.exists(result["path"])
    assert await async_read_bytes(result["path"]) == data


@pytest.mark.anyio
async def test_save_attachment_sanitization(temp_sessions_dir):
    manager = AttachmentManager(sessions_dir=temp_sessions_dir)
    filename = "test/../file!@#$.txt"
    workspace = "test_ws"
    data = b"hello world"

    result = await manager.save_attachment(filename=filename, workspace_name=workspace, data=data)

    assert result["filename"] == "test..file.txt"
    assert os.path.exists(result["path"])


@pytest.mark.anyio
async def test_save_attachment_empty_filename(temp_sessions_dir):
    manager = AttachmentManager(sessions_dir=temp_sessions_dir)
    filename = "!@#$"
    workspace = "test_ws"
    data = b"hello world"

    result = await manager.save_attachment(filename=filename, workspace_name=workspace, data=data, collision_id="123")

    assert result["filename"] == "attachment_123"
    assert os.path.exists(result["path"])


@pytest.mark.anyio
async def test_save_attachment_collision(temp_sessions_dir):
    manager = AttachmentManager(sessions_dir=temp_sessions_dir)
    filename = "test.txt"
    workspace = "test_ws"
    data1 = b"first"
    data2 = b"second"

    # Save first file
    result1 = await manager.save_attachment(filename=filename, workspace_name=workspace, data=data1)

    # Save second file with same name
    result2 = await manager.save_attachment(filename=filename, workspace_name=workspace, data=data2, collision_id="new")

    assert result1["filename"] == "test.txt"
    assert result2["filename"] == "test_new.txt"
    assert os.path.exists(result1["path"])
    assert os.path.exists(result2["path"])
    assert await async_read_bytes(result1["path"]) == data1
    assert await async_read_bytes(result2["path"]) == data2


@pytest.mark.anyio
async def test_save_attachment_with_callback(temp_sessions_dir):
    manager = AttachmentManager(sessions_dir=temp_sessions_dir)
    filename = "test_cb.txt"
    workspace = "test_ws"
    data = b"callback data"

    async def save_cb(path):
        with open(path, "wb") as f:
            f.write(data)

    result = await manager.save_attachment(filename=filename, workspace_name=workspace, save_callback=save_cb)

    assert result["filename"] == "test_cb.txt"
    assert os.path.exists(result["path"])
    assert await async_read_bytes(result["path"]) == data


@pytest.mark.anyio
async def test_save_attachment_invalid_arguments(temp_sessions_dir):
    manager = AttachmentManager(sessions_dir=temp_sessions_dir)
    filename = "test.txt"
    workspace = "test_ws"

    with pytest.raises(ValueError, match="Either 'data' or 'save_callback' must be provided"):
        await manager.save_attachment(filename=filename, workspace_name=workspace)
