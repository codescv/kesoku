import os
import sqlite3

from kesoku.config import get_config, load_config


def migrate() -> None:
    """Migrate database user_preferences category memories into roles/*/preferences.md files."""
    config_path = "config.toml"
    if os.path.exists(config_path):
        load_config(config_path)
    cfg = get_config()
    db_path = cfg.workspace.db_path
    roles_dir = cfg.workspace.roles_dir
    if not os.path.isabs(db_path) and cfg.agent_working_dir:
        db_path = os.path.join(cfg.agent_working_dir, db_path)
    if not os.path.isabs(roles_dir) and cfg.agent_working_dir:
        roles_dir = os.path.join(cfg.agent_working_dir, roles_dir)

    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}. Nothing to migrate.")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='agent_memories'")
        if not cursor.fetchone():
            print("Table 'agent_memories' does not exist in the database. Nothing to migrate.")
            return

        cursor.execute("SELECT * FROM agent_memories WHERE category = 'user_preferences'")
        rows = cursor.fetchall()
        if not rows:
            print("No user_preferences records found in the database.")
            return

        print(f"Found {len(rows)} user_preferences records to migrate.")
        for row in rows:
            role = row["role"]
            content = row["content"]

            role_dir = os.path.join(roles_dir, role)
            os.makedirs(role_dir, exist_ok=True)

            pref_path = os.path.join(role_dir, "preferences.md")

            file_exists = os.path.exists(pref_path)
            with open(pref_path, "a", encoding="utf-8") as f:
                if file_exists:
                    f.write("\n\n")
                f.write(content)

            print(f"Migrated memory key '{row['key']}' for role '{role}' to {pref_path}")

            cursor.execute(
                "DELETE FROM agent_memories WHERE id = ?",
                (row["id"],)
            )

        conn.commit()
        print("Migration completed successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
