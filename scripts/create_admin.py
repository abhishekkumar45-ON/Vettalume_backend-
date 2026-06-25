"""Create or promote a Vettalume content admin.

This is the SECURE bootstrap: it runs on the server (it touches the database directly), so there is
no public "make me an admin" endpoint to exploit. Use it to mint the first admin; after that, that
admin can grant others from the portal (Admins tab) or via POST /admin/admins.

    python -m scripts.create_admin <email> <password> [display_name]
    # e.g.
    python -m scripts.create_admin you@yourco.com "a-strong-password" "Abhishek"

If the email already has an account, its password is reset and it is granted admin.
"""
from __future__ import annotations

import sys
import uuid

from sqlalchemy import select

from app import models
from app.db import SessionLocal, init_db
from app.services import security


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: python -m scripts.create_admin <email> <password> [display_name]")
        sys.exit(1)
    email = sys.argv[1].strip().lower()
    password = sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else "Admin"
    if len(password) < 8:
        print("error: password must be at least 8 characters")
        sys.exit(1)

    init_db()
    db = SessionLocal()
    try:
        acc = db.scalar(select(models.Account).where(models.Account.email == email))
        if acc is None:
            acc = models.Account(id=uuid.uuid4(), email=email, display_name=name)
            db.add(acc)
            db.flush()
            db.add(models.Credential(account_id=acc.id, password_hash=security.hash_password(password)))
            print(f"created account  {email}")
        else:
            cred = db.get(models.Credential, acc.id)
            if cred is None:
                db.add(models.Credential(account_id=acc.id, password_hash=security.hash_password(password)))
            else:
                cred.password_hash = security.hash_password(password)
            print(f"account exists   {email}  (password reset)")

        if db.get(models.AdminUser, acc.id) is None:
            db.add(models.AdminUser(account_id=acc.id, role="admin"))
            print("granted admin    ✓")
        else:
            print("already admin    ✓")

        db.commit()
        print(f"\n✅ {email} is now a content admin.")
        print("   Open  http://localhost:8001/admin  and log in with this email + password.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
