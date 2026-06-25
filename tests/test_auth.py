"""Phase 6 — real auth: password hashing, HS256 JWT, and the register/login/me HTTP flow."""
from fastapi.testclient import TestClient

from app.main import app
from app.services import security


# ---------------- password hashing ----------------
def test_password_hash_roundtrip():
    h = security.hash_password("hunter2pass")
    assert h.startswith("pbkdf2_sha256$")
    assert security.verify_password("hunter2pass", h) is True
    assert security.verify_password("wrongpass", h) is False


def test_verify_rejects_malformed_hash():
    assert security.verify_password("x", "not-a-valid-hash") is False


# ---------------- JWT ----------------
def test_jwt_roundtrip_carries_subject():
    tok = security.make_token("user-123")
    assert security.decode_token(tok)["sub"] == "user-123"


def test_jwt_expired_is_rejected():
    tok = security.make_token("u", expires_in=-5)
    try:
        security.decode_token(tok)
        assert False, "expired token should raise"
    except ValueError as e:
        assert "expired" in str(e)


def test_jwt_tampered_signature_is_rejected():
    tok = security.make_token("u")
    try:
        security.decode_token(tok[:-3] + "AAA")
        assert False, "tampered token should raise"
    except ValueError:
        pass


def test_jwt_alg_confusion_is_rejected():
    # forge a token whose header says alg=none but reuse a valid signature segment
    import json
    from app.services.security import _b64u_encode, _sign
    header = _b64u_encode(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    payload = _b64u_encode(json.dumps({"sub": "evil", "exp": 9999999999}).encode())
    sig = _b64u_encode(_sign(f"{header}.{payload}".encode()))
    try:
        security.decode_token(f"{header}.{payload}.{sig}")
        assert False, "alg=none must be rejected"
    except ValueError as e:
        assert "alg" in str(e)


# ---------------- HTTP flow ----------------
def test_register_login_me_flow():
    with TestClient(app) as c:
        reg = c.post("/auth/register",
                     json={"email": "p6reg@x.com", "password": "hunter2pass"})
        assert reg.status_code == 200
        tok = reg.json()["access_token"]
        me = c.get("/auth/me", headers={"Authorization": f"Bearer {tok}"})
        assert me.status_code == 200 and me.json()["email"] == "p6reg@x.com"

        assert c.post("/auth/register",
                      json={"email": "p6reg@x.com", "password": "otherpass1"}).status_code == 409
        assert c.post("/auth/register",
                      json={"email": "p6short@x.com", "password": "short"}).status_code == 400

        assert c.post("/auth/login",
                      json={"email": "p6reg@x.com", "password": "nope"}).status_code == 401
        good = c.post("/auth/login", json={"email": "p6reg@x.com", "password": "hunter2pass"})
        assert good.status_code == 200 and good.json()["token_type"] == "bearer"


def test_me_requires_auth_and_rejects_bad_token():
    with TestClient(app) as c:
        assert c.get("/auth/me").status_code == 401
        assert c.get("/auth/me",
                     headers={"Authorization": "Bearer not.a.token"}).status_code == 401


def test_legacy_x_learner_id_still_works():
    with TestClient(app) as c:
        lid = c.post("/auth/dev-login", json={"email": "p6legacy@x.com"}).json()["learner_id"]
        me = c.get("/auth/me", headers={"X-Learner-Id": lid})
        assert me.status_code == 200 and me.json()["email"] == "p6legacy@x.com"
