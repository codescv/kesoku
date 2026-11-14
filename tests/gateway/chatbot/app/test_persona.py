"""Tests for the persona / user profile / avatar restyle service (`app/persona.py`)."""

import io
import json
import os
import sys
import textwrap
import time
import uuid
import wave

import pytest
from PIL import Image

from kesoku.config import AppConfig, KesokuConfig
from kesoku.db.manager import DatabaseManager
from kesoku.gateway.chatbot.app import persona as persona_mod
from kesoku.gateway.chatbot.app.host import Reply, RequestContext
from kesoku.gateway.chatbot.app.persona import PersonaService, fill_argv_template, sanitize_prompt_hint
from kesoku.gateway.chatbot.app.protocol import ProtocolError

DEVICE = "dev_1"


def _png_bytes(size=(64, 48), color=(200, 30, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


class FakeMedia:
    def __init__(self):
        self.objects = {}
        self.uploads = []

    def put(self, data, *, device_id=DEVICE, issued_at=None, ext="png"):
        obj = f"media/avatar/{uuid.uuid4()}.{ext}"
        self.objects[obj] = {
            "data": data,
            "issued_device_id": device_id,
            "issued_at": time.time() if issued_at is None else issued_at,
        }
        return obj

    async def sign(self, object_name):
        meta = self.objects.get(object_name)
        if meta is None:
            raise KeyError(object_name)
        return {
            "object": object_name,
            "url": f"https://signed.invalid/{object_name}",
            "expires_at": time.time() + 900,
            "issued_device_id": meta["issued_device_id"],
            "issued_at": meta["issued_at"],
        }

    async def download(self, url, max_bytes):
        obj = url.split("https://signed.invalid/", 1)[1]
        data = self.objects[obj]["data"]
        if len(data) > max_bytes:
            raise ValueError("too large")
        return data

    async def upload_file(self, path, *, purpose, session_id=None, mime=None, index=None):
        with open(path, "rb") as f:
            data = f.read()
        suffix = f"-cand{index}" if index is not None else ""
        obj = f"media/avatar/{uuid.uuid4()}{suffix}.webp" if purpose == "avatar_candidate" else (
            f"media/asset/{uuid.uuid4()}.wav"
        )
        self.uploads.append({"path": path, "purpose": purpose, "mime": mime, "index": index, "data": data})
        return {"object": obj, "url": f"https://signed.invalid/{obj}", "expires_at": 1.0, "mime": mime,
                "bytes": len(data)}


class FakeHost:
    def __init__(self, tmp_path, app_config):
        self.chatbot_id = "app"
        self.app_config = app_config
        self.media = FakeMedia()
        self._awd = str(tmp_path / "awd")
        os.makedirs(self._awd, exist_ok=True)
        self._cfg = KesokuConfig()
        self._cfg.workspace.skills_dir = str(tmp_path / "no_skills")
        self._db = DatabaseManager(str(tmp_path / "k.db"))
        self._db.init_tables()
        self.broadcasts = []
        self.unicasts = []

    @property
    def kconfig(self):
        return self._cfg

    @property
    def sync_db(self):
        return self._db

    @property
    def awd(self):
        return self._awd

    @property
    def roles_dir(self):
        return os.path.join(self._awd, "roles")

    @property
    def sessions_dir(self):
        return os.path.join(self._awd, "sessions")

    async def broadcast(self, event_type, payload):
        self.broadcasts.append((event_type, payload))

    async def send_to_device(self, device_id, event_type, payload, *, reply_to=None, session_id=None):
        self.unicasts.append((device_id, event_type, payload))

    async def emit_session_event(self, session_id, event_type, payload):
        return 0


GEN_SYNC = textwrap.dedent(
    """
    import sys
    from PIL import Image
    source, output, style, prompt = sys.argv[1:5]
    assert Image.open(source).format == "PNG"
    Image.new("RGB", (300, 200), (10, 120, 200)).save(output, "PNG")
    """
)

GEN_FAIL = textwrap.dedent(
    """
    import sys
    sys.stderr.write("model quota exhausted API_KEY=abc123\\n")
    sys.exit(3)
    """
)

GEN_ERROR_SIDECAR = textwrap.dedent(
    """
    import json, sys
    output = sys.argv[2]
    json.dump({"job_id": "j", "error": "blocked by safety filter"}, open(output + ".error.json", "w"))
    """
)

# Two-phase producer: install `.pending.json`, spawn a detached worker that writes the target later.
GEN_ASYNC = textwrap.dedent(
    """
    import json, os, subprocess, sys, time
    output = sys.argv[2]
    worker = (
        "import os, sys, time, json\\n"
        "from PIL import Image\\n"
        "out = sys.argv[1]\\n"
        "time.sleep(0.4)\\n"
        "Image.new('RGB', (256, 256), (0, 200, 0)).save(out + '.part.png', 'PNG')\\n"
        "os.replace(out + '.part.png', out)\\n"
        "os.unlink(out + '.pending.json')\\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", worker, output], start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    marker = {"job_id": "j1", "created_at": time.time(), "timeout_sec": 60, "pid": proc.pid}
    with open(output + ".pending.json", "w") as f:
        json.dump(marker, f)
    """
)


def _gen_cmd(tmp_path, body, name):
    script = tmp_path / f"{name}.py"
    script.write_text(body, encoding="utf-8")
    return [sys.executable, str(script), "{source}", "{output}", "{style}", "{prompt}"]


@pytest.fixture
async def make_service(tmp_path):
    created = []

    def _make(generator=GEN_SYNC, voice_cmd=None):
        cfg = AppConfig(
            avatar_generator_command=_gen_cmd(tmp_path, generator, f"gen_{len(created)}") if generator else [],
            voice_preview_command=voice_cmd or [],
        )
        host = FakeHost(tmp_path, cfg)
        roles = host.roles_dir
        os.makedirs(os.path.join(roles, "default"), exist_ok=True)
        with open(os.path.join(roles, "default", "intro.md"), "w", encoding="utf-8") as f:
            f.write("# Name\nDefault Helper\n")
        svc = PersonaService(host, watch_timeout_sec=20.0, poll_min_sec=0.05, poll_max_sec=0.2)
        created.append(svc)
        return svc, host

    yield _make
    for svc in created:
        await svc.stop()


def _ctx(request_type="x", device_id=DEVICE):
    return RequestContext(device_id=device_id, request_id="r1", request_type=request_type)


async def _expect_error(code, coro):
    with pytest.raises(ProtocolError) as ei:
        await coro
    assert ei.value.code == code, ei.value.message
    return ei.value


# ---------------------------------------------------------------- helpers


def test_sanitize_prompt_hint_and_template_single_pass():
    assert sanitize_prompt_hint("--style evil\nline\x07two") == "style evil line two"
    assert len(sanitize_prompt_hint("a" * 200)) == 80
    out = fill_argv_template(["--prompt", "{prompt}", "{output}"], {"prompt": "{output}", "output": "/o.png"})
    assert out == ["--prompt", "{output}", "/o.png"]


# ---------------------------------------------------------------- persona.*


async def test_persona_list_get_and_create(make_service):
    svc, host = make_service()
    await svc.handle_user_profile_put(_ctx(), {"nickname": "Sam", "bio": "", "cameo_opt_in": False})
    host.broadcasts.clear()

    listed = await svc.handle_persona_list(_ctx(), {})
    assert [r["role_id"] for r in listed["roles"]] == ["default"]
    assert listed["roles"][0]["builtin"] is True
    assert listed["roles"][0]["display_name"] == "Default Helper"

    res = await svc.handle_persona_upsert(
        _ctx(),
        {"display_name": "Night Owl", "tagline": "late shift", "intro_md": "# Name\nNight Owl\n",
         "preferences_md": "be brief", "voice_preset": "calm"},
    )
    role = res["role"]
    assert res["ok"] is True
    assert role["role_id"] == "night_owl" and role["builtin"] is False and role["voice_preset"] == "calm"
    assert host.broadcasts[-1] == ("persona.updated", {"roles": [role]})
    # New roles get the user profile block seeded into their pinned memory.
    prof = os.path.join(host.roles_dir, "night_owl", "memories", "user", "profile.md")
    assert "称呼：Sam" in open(prof, encoding="utf-8").read()

    again = await svc.handle_persona_upsert(_ctx(), {"display_name": "Night Owl", "intro_md": "", "preferences_md": ""})
    assert again["role"]["role_id"] == "night_owl_2"

    detail = await svc.handle_persona_get(_ctx(), {"role_id": "night_owl"})
    assert detail["intro_md"] == "# Name\nNight Owl\n" and detail["preferences_md"] == "be brief"
    await _expect_error("NOT_FOUND", svc.handle_persona_get(_ctx(), {"role_id": "missing"}))
    await _expect_error("BAD_REQUEST", svc.handle_persona_get(_ctx(), {"role_id": "../x"}))


async def test_persona_upsert_validation_and_builtin_update(make_service):
    svc, host = make_service()
    big = "x" * (64 * 1024 + 1)
    await _expect_error("BAD_REQUEST", svc.handle_persona_upsert(_ctx(), {"display_name": "A" * 33}))
    await _expect_error("BAD_REQUEST", svc.handle_persona_upsert(_ctx(), {"display_name": "A", "tagline": "t" * 81}))
    await _expect_error("BAD_REQUEST", svc.handle_persona_upsert(_ctx(), {"display_name": "A", "intro_md": big}))
    await _expect_error("BAD_REQUEST", svc.handle_persona_upsert(_ctx(), {"display_name": "A", "role_id": "Bad/Id"}))
    await _expect_error(
        "PERSONA_BUILTIN",
        svc.handle_persona_upsert(_ctx(), {"display_name": "A", "role_id": "default", "new_role_id": "other"}),
    )

    res = await svc.handle_persona_upsert(
        _ctx(), {"role_id": "default", "display_name": "Renamed Label", "intro_md": "new intro", "preferences_md": "p"}
    )
    assert res["role"]["builtin"] is True
    assert res["role"]["display_name"] == "Renamed Label"
    assert open(os.path.join(host.roles_dir, "default", "intro.md"), encoding="utf-8").read() == "new intro"


async def test_persona_upsert_with_direct_avatar(make_service):
    svc, host = make_service()
    obj = host.media.put(_png_bytes((900, 300)))
    res = await svc.handle_persona_upsert(_ctx(), {"display_name": "Alice", "avatar_object": obj})
    role = res["role"]
    assert role["avatar_rev"] == 1 and role["avatar_ai"] is False
    assert role["avatar_ref"] == "roles/alice/avatar.webp"
    with Image.open(os.path.join(host.roles_dir, "alice", "avatar.webp")) as im:
        assert im.format == "WEBP" and im.size == (512, 512)
    assert svc.resolve_avatar_ref("roles/alice/avatar.webp").endswith("alice/avatar.webp")
    assert svc.resolve_avatar_ref("roles/../x/avatar.webp") is None
    assert svc.resolve_avatar_ref("user/avatar.webp") is None


async def test_persona_delete_moves_to_trash_and_reassigns(make_service, monkeypatch):
    svc, host = make_service()
    await svc.handle_persona_upsert(_ctx(), {"display_name": "Temp Role", "role_id": "temp_role"})
    chars = os.path.join(svc.characters_dir, "temp_role")
    os.makedirs(chars)
    calls = []
    orig = host.sync_db.reassign_role_references
    monkeypatch.setattr(host.sync_db, "reassign_role_references", lambda a, b: calls.append((a, b)) or orig(a, b))

    await _expect_error("PERSONA_BUILTIN", svc.handle_persona_delete(_ctx(), {"role_id": "default"}))
    await _expect_error("NOT_FOUND", svc.handle_persona_delete(_ctx(), {"role_id": "nobody"}))
    res = await svc.handle_persona_delete(_ctx(), {"role_id": "temp_role"})

    assert res == {"ok": True}
    assert calls == [("temp_role", "default")]
    assert not os.path.exists(os.path.join(host.roles_dir, "temp_role"))
    trash = os.listdir(os.path.join(host.roles_dir, ".trash"))
    assert len(trash) == 1 and trash[0].startswith("temp_role-")
    assert os.listdir(os.path.join(svc.characters_dir, ".trash"))[0].startswith("temp_role-")
    assert not os.path.exists(chars)
    assert host.broadcasts[-1] == ("persona.updated", {"roles": [], "deleted": ["temp_role"]})
    listed = await svc.handle_persona_list(_ctx(), {})
    assert [r["role_id"] for r in listed["roles"]] == ["default"]


# ---------------------------------------------------------------- user_profile.*


async def test_user_profile_put_syncs_blocks_and_broadcasts(make_service):
    svc, host = make_service()
    await svc.handle_persona_upsert(_ctx(), {"display_name": "Alice", "role_id": "alice"})
    await _expect_error("BAD_REQUEST", svc.handle_user_profile_put(_ctx(), {"nickname": "n" * 33}))
    await _expect_error("BAD_REQUEST", svc.handle_user_profile_put(_ctx(), {"bio": "b" * 2049}))
    await _expect_error("BAD_REQUEST", svc.handle_user_profile_put(_ctx(), {"cameo_opt_in": "yes"}))

    res = await svc.handle_user_profile_put(_ctx(), {"nickname": "Sam", "bio": "Builds things", "cameo_opt_in": False})

    assert res["ok"] is True and res["profile"]["nickname"] == "Sam"
    assert host.broadcasts[-1] == ("user_profile.updated", res["profile"])
    for rid in ("default", "alice"):
        text = open(os.path.join(host.roles_dir, rid, "memories", "user", "profile.md"), encoding="utf-8").read()
        assert "称呼：Sam" in text and "简介：Builds things" in text
    got = await svc.handle_user_profile_get(_ctx(), {})
    assert got["nickname"] == "Sam" and got["cameo_opt_in"] is False and got["avatar_rev"] == 0
    with open(os.path.join(host.awd, "user", "profile.json"), encoding="utf-8") as f:
        assert json.load(f)["bio"] == "Builds things"


# ---------------------------------------------------------------- avatar.generate


async def test_avatar_generate_source_validation(make_service, monkeypatch):
    svc, host = make_service()
    base = {"target_type": "user", "target_id": "me", "style_preset": "anime_clean", "count": 4}

    await _expect_error("AVATAR_SOURCE_INVALID", svc.handle_avatar_generate(_ctx(), {**base, "source_object": "x"}))
    unknown = f"media/avatar/{uuid.uuid4()}.png"
    await _expect_error("AVATAR_SOURCE_INVALID", svc.handle_avatar_generate(_ctx(), {**base, "source_object": unknown}))
    other = host.media.put(_png_bytes(), device_id="dev_other")
    await _expect_error("AVATAR_SOURCE_INVALID", svc.handle_avatar_generate(_ctx(), {**base, "source_object": other}))
    stale = host.media.put(_png_bytes(), issued_at=time.time() - 700)
    await _expect_error("AVATAR_SOURCE_INVALID", svc.handle_avatar_generate(_ctx(), {**base, "source_object": stale}))
    junk = host.media.put(b"GIF89a" + b"\x00" * 100)
    await _expect_error("AVATAR_SOURCE_INVALID", svc.handle_avatar_generate(_ctx(), {**base, "source_object": junk}))
    fake_png = host.media.put(b"\x89PNG\r\n\x1a\n" + b"garbage" * 10)
    await _expect_error(
        "AVATAR_SOURCE_INVALID", svc.handle_avatar_generate(_ctx(), {**base, "source_object": fake_png})
    )
    frames = [Image.new("RGB", (16, 16), c) for c in ((255, 0, 0), (0, 255, 0))]
    buf = io.BytesIO()
    frames[0].save(buf, "WEBP", save_all=True, append_images=frames[1:])
    animated = host.media.put(buf.getvalue(), ext="webp")
    await _expect_error(
        "AVATAR_SOURCE_INVALID", svc.handle_avatar_generate(_ctx(), {**base, "source_object": animated})
    )
    monkeypatch.setattr(persona_mod, "AVATAR_MAX_PIXELS", 100)
    huge = host.media.put(_png_bytes((20, 20)))
    await _expect_error("AVATAR_SOURCE_INVALID", svc.handle_avatar_generate(_ctx(), {**base, "source_object": huge}))

    ok_obj = host.media.put(_png_bytes())
    await _expect_error(
        "BAD_REQUEST", svc.handle_avatar_generate(_ctx(), {**base, "source_object": ok_obj, "style_preset": "nope"})
    )
    await _expect_error(
        "NOT_FOUND",
        svc.handle_avatar_generate(_ctx(), {**base, "source_object": ok_obj, "target_type": "role",
                                            "target_id": "ghost"}),
    )
    assert host.unicasts == []


async def test_avatar_generate_rate_limited(make_service):
    svc, host = make_service()
    now = time.time()
    for i in range(6):
        host.sync_db.insert_avatar_job(
            job_id=f"old{i}", device_id=DEVICE, target_type="user", target_id="user", style_preset="pixel",
            prompt_hint=None, source_path="/nonexistent", created_at=now - 60,
        )
    obj = host.media.put(_png_bytes())
    await _expect_error(
        "RATE_LIMITED",
        svc.handle_avatar_generate(_ctx(), {"target_type": "user", "target_id": "me", "source_object": obj,
                                            "style_preset": "pixel"}),
    )
    # Another device is unaffected.
    obj2 = host.media.put(_png_bytes(), device_id="dev_2")
    reply = await svc.handle_avatar_generate(
        _ctx(device_id="dev_2"),
        {"target_type": "user", "target_id": "me", "source_object": obj2, "style_preset": "pixel"},
    )
    assert reply.type == "avatar.job"
    await svc.wait_idle()


async def test_avatar_generate_without_generator_is_bad_request(make_service):
    svc, host = make_service(generator=None)
    obj = host.media.put(_png_bytes())
    await _expect_error(
        "BAD_REQUEST",
        svc.handle_avatar_generate(_ctx(), {"target_type": "user", "target_id": "me", "source_object": obj,
                                            "style_preset": "pixel"}),
    )


async def _generate(svc, host, target_type="role", target_id="alice"):
    obj = host.media.put(_png_bytes((640, 480)))
    reply = await svc.handle_avatar_generate(
        _ctx("avatar.generate"),
        {"target_type": target_type, "target_id": target_id, "source_object": obj, "style_preset": "watercolor",
         "prompt_hint": "--wear a red scarf\n", "count": 4},
    )
    await svc.wait_idle()
    return reply


async def test_avatar_generate_happy_path_and_apply_role(make_service):
    svc, host = make_service()
    await svc.handle_persona_upsert(_ctx(), {"display_name": "Alice", "role_id": "alice"})
    reply = await _generate(svc, host)

    assert isinstance(reply, Reply) and reply.type == "avatar.job"
    job_id = reply.payload["job_id"]
    assert len(reply.payload["placeholders"]) == 4 and reply.payload["eta_ms"] > 0
    ready = [p for d, t, p in host.unicasts if t == "media.ready"]
    assert len(ready) == 4 and all(d == DEVICE for d, _, _ in host.unicasts)
    assert sorted(p["index"] for p in ready) == [0, 1, 2, 3]
    assert {p["placeholder_id"] for p in ready} == set(reply.payload["placeholders"])
    assert all(p["kind"] == "avatar_candidate" and p["job_id"] == job_id for p in ready)
    assert all(u["purpose"] == "avatar_candidate" for u in host.media.uploads)
    with Image.open(io.BytesIO(host.media.uploads[0]["data"])) as im:
        assert im.format == "WEBP" and im.size == (512, 512)
    job = host.sync_db.get_avatar_job(job_id)
    assert job["status"] == "done"
    assert all(c["status"] == "ready" for c in job["candidates"])

    # Existing registry entry: image/aliases updated, gm fields preserved.
    cdir = os.path.join(svc.characters_dir, "alice")
    os.makedirs(cdir)
    entry = {"name": "Alice Example", "aliases": ["Alice"], "image": "roles/alice/images/old.jpg",
             "trigger": "a librarian", "draftable": True, "hierarchy": {"rank": 1}}
    with open(os.path.join(cdir, "characters.json"), "w", encoding="utf-8") as f:
        json.dump([entry], f)

    await _expect_error(
        "NOT_FOUND",
        svc.handle_avatar_apply(_ctx(device_id="dev_x"), {"job_id": job_id, "index": 1, "target_type": "role",
                                                          "target_id": "alice", "register_reference": True}),
    )
    await _expect_error(
        "BAD_REQUEST",
        svc.handle_avatar_apply(_ctx(), {"job_id": job_id, "index": 1, "target_type": "role",
                                         "target_id": "default", "register_reference": False}),
    )
    host.broadcasts.clear()
    res = await svc.handle_avatar_apply(
        _ctx(), {"job_id": job_id, "index": 1, "target_type": "role", "target_id": "alice",
                 "register_reference": True}
    )

    assert res == {"ok": True, "avatar_rev": 1}
    event, payload = host.broadcasts[-1]
    assert event == "persona.updated" and payload["roles"][0]["avatar_ai"] is True
    with Image.open(os.path.join(host.roles_dir, "alice", "avatar.webp")) as im:
        assert im.size == (512, 512)
    assert os.path.isfile(os.path.join(cdir, "images", "ref_face.webp"))
    with open(os.path.join(cdir, "characters.json"), encoding="utf-8") as f:
        entries = json.load(f)
    assert len(entries) == 1
    e = entries[0]
    assert e["image"] == "characters/alice/images/ref_face.webp"
    assert e["trigger"] == "a librarian" and e["draftable"] is True and e["hierarchy"] == {"rank": 1}
    assert e["aliases"] == ["Alice", "alice"]
    with open(os.path.join(cdir, ".kesoku_prev_image"), encoding="utf-8") as f:
        assert json.load(f)["image"] == "roles/alice/images/old.jpg"

    res2 = await svc.handle_avatar_apply(
        _ctx(), {"job_id": job_id, "index": 2, "target_type": "role", "target_id": "alice",
                 "register_reference": True}
    )
    assert res2["avatar_rev"] == 2
    with open(os.path.join(cdir, ".kesoku_prev_image"), encoding="utf-8") as f:
        assert json.load(f)["image"] == "roles/alice/images/old.jpg"  # original kept


async def test_avatar_apply_user_registers_and_cameo_off_restores(make_service):
    svc, host = make_service()
    await svc.handle_user_profile_put(_ctx(), {"nickname": "Sam", "bio": "", "cameo_opt_in": False})
    reply = await _generate(svc, host, target_type="user", target_id="me")
    job_id = reply.payload["job_id"]

    res = await svc.handle_avatar_apply(
        _ctx(), {"job_id": job_id, "index": 0, "target_type": "user", "target_id": "me", "register_reference": True}
    )

    assert res["avatar_rev"] == 1
    event, view = host.broadcasts[-1]
    assert event == "user_profile.updated" and view["cameo_opt_in"] is True and view["avatar_ref"] == "user/avatar.webp"
    reg = os.path.join(svc.characters_dir, "user", "characters.json")
    with open(reg, encoding="utf-8") as f:
        entries = json.load(f)
    assert entries == [{"name": "Sam", "aliases": [], "image": "characters/user/images/ref_face.webp",
                        "draftable": False}]
    assert svc.resolve_avatar_ref("user/avatar.webp")

    await svc.handle_user_profile_put(_ctx(), {"nickname": "Sam", "bio": "", "cameo_opt_in": False})
    with open(reg, encoding="utf-8") as f:
        assert json.load(f)[0]["image"] == ""
    assert not os.path.exists(os.path.join(svc.characters_dir, "user", ".kesoku_prev_image"))

    await svc.handle_user_profile_put(_ctx(), {"nickname": "Sam", "bio": "", "cameo_opt_in": True})
    with open(reg, encoding="utf-8") as f:
        assert json.load(f)[0]["image"] == "characters/user/images/ref_face.webp"


async def test_avatar_generate_failures_emit_media_failed(make_service):
    svc, host = make_service(generator=GEN_FAIL)
    reply = await _generate(svc, host, target_type="user", target_id="me")
    failed = [p for _, t, p in host.unicasts if t == "media.failed"]
    assert len(failed) == 4
    assert all("abc123" not in p["reason"] for p in failed)
    assert host.sync_db.get_avatar_job(reply.payload["job_id"])["status"] == "failed"
    await _expect_error(
        "BAD_REQUEST",
        svc.handle_avatar_apply(_ctx(), {"job_id": reply.payload["job_id"], "index": 0, "target_type": "user",
                                         "target_id": "me", "register_reference": False}),
    )


async def test_avatar_generate_error_sidecar(make_service):
    svc, host = make_service(generator=GEN_ERROR_SIDECAR)
    await _generate(svc, host, target_type="user", target_id="me")
    failed = [p for _, t, p in host.unicasts if t == "media.failed"]
    assert len(failed) == 4 and all("safety" in p["reason"] for p in failed)


async def test_avatar_generate_two_phase_async_producer(make_service):
    svc, host = make_service(generator=GEN_ASYNC)
    await _generate(svc, host, target_type="user", target_id="me")
    ready = [p for _, t, p in host.unicasts if t == "media.ready"]
    assert len(ready) == 4, host.unicasts


async def test_cleanup_expired_jobs(make_service):
    svc, host = make_service()
    reply = await _generate(svc, host, target_type="user", target_id="me")
    job_id = reply.payload["job_id"]
    job_dir = os.path.join(svc.jobs_root, job_id)
    assert os.path.isdir(job_dir)
    with host.sync_db.connection_provider.connection() as conn, conn:
        conn.execute("UPDATE avatar_jobs SET expires_at = ? WHERE job_id = ?", (time.time() - 1, job_id))
    assert await svc.cleanup_expired_jobs() == 1
    assert not os.path.exists(job_dir)
    await _expect_error(
        "NOT_FOUND",
        svc.handle_avatar_apply(_ctx(), {"job_id": job_id, "index": 0, "target_type": "user", "target_id": "me",
                                         "register_reference": False}),
    )


# ---------------------------------------------------------------- voice.preview


async def test_voice_preview(make_service, tmp_path):
    svc, host = make_service()
    await _expect_error("BAD_REQUEST", svc.handle_voice_preview(_ctx(), {"voice_preset": "calm"}))

    script = tmp_path / "tts.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys, wave
            voice, text, out = sys.argv[1:4]
            assert voice == "calm" and text
            with wave.open(out, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
                w.writeframes(b"\\x00\\x00" * 8000)
            """
        ),
        encoding="utf-8",
    )
    svc2, host2 = make_service(voice_cmd=[sys.executable, str(script), "{voice}", "{text}", "{output}"])
    reply = await svc2.handle_voice_preview(_ctx(), {"voice_preset": "calm", "text": "hello"})

    assert reply.type == "media.ready"
    assert reply.payload["kind"] == "voice" and reply.payload["mime"] == "audio/wav"
    assert reply.payload["duration_ms"] == 1000
    assert host2.media.uploads[-1]["purpose"] == "asset"
    assert not os.path.exists(host2.media.uploads[-1]["path"])  # local temp removed
    with wave.open(io.BytesIO(host2.media.uploads[-1]["data"])) as w:
        assert w.getnframes() == 8000


async def test_start_stop_lifecycle(make_service):
    svc, host = make_service()
    await svc.start()
    await svc.stop()
