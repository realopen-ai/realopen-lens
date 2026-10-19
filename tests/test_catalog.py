"""Catalog format upgrades preserve existing collection analyses."""

from realopen_lens.catalog import Catalog


def test_legacy_catalog_migration_preserves_analysis_records(tmp_path):
    import sqlite3

    connection = sqlite3.connect(tmp_path / "catalog.sqlite3")
    connection.executescript("""
        CREATE TABLE analyses (
            pipeline TEXT, image_id TEXT, stage TEXT, status TEXT, payload TEXT,
            artifact TEXT, checksum TEXT, PRIMARY KEY(pipeline,image_id,stage));
        INSERT INTO analyses VALUES ('p','i','ocr','error','{}',NULL,NULL);
        PRAGMA user_version=1;
    """)
    connection.close()
    catalog = Catalog(tmp_path, readonly=True)
    assert catalog.analysis("p", "i", "ocr")["status"] == "error"
    catalog.close()
    catalog = Catalog(tmp_path)
    assert catalog.db.execute("PRAGMA user_version").fetchone()[0] == 2
    assert catalog.analysis("p", "i", "ocr")["job_id"] is None
    catalog.save_analysis("p", "i", "ocr", {"status": "ok", "text": "recovered"}, job_id="new")
    assert catalog.analysis("p", "i", "ocr")["job_id"] == "new"
    catalog.close()
