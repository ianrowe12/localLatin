import argparse
import os
from pathlib import Path
from typing import Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def build_drive_service(credentials_path: str):
    creds = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def find_folder(service, name: str, parent_id: Optional[str] = None) -> Optional[str]:
    q = [
        "mimeType='application/vnd.google-apps.folder'",
        f"name='{name}'",
        "trashed=false",
    ]
    if parent_id:
        q.append(f"'{parent_id}' in parents")
    resp = (
        service.files()
        .list(q=" and ".join(q), fields="files(id, name)", pageSize=10)
        .execute()
    )
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def create_folder(service, name: str, parent_id: Optional[str] = None) -> str:
    metadata = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        metadata["parents"] = [parent_id]
    folder = service.files().create(body=metadata, fields="id").execute()
    return folder["id"]


def get_or_create_folder(service, name: str, parent_id: Optional[str] = None) -> str:
    folder_id = find_folder(service, name, parent_id)
    if folder_id:
        return folder_id
    return create_folder(service, name, parent_id)


def find_file(service, name: str, parent_id: str) -> Optional[str]:
    q = [
        f"name='{name}'",
        "trashed=false",
        f"'{parent_id}' in parents",
    ]
    resp = (
        service.files()
        .list(q=" and ".join(q), fields="files(id, name)", pageSize=10)
        .execute()
    )
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def upload_file(service, local_path: Path, parent_id: str) -> str:
    file_id = find_file(service, local_path.name, parent_id)
    media = MediaFileUpload(local_path.as_posix(), resumable=True)
    if file_id:
        updated = service.files().update(fileId=file_id, media_body=media).execute()
        return updated["id"]
    metadata = {"name": local_path.name, "parents": [parent_id]}
    created = (
        service.files().create(body=metadata, media_body=media, fields="id").execute()
    )
    return created["id"]


def sync_run_dir(
    local_run_dir: Path,
    credentials_path: str,
    drive_root_folder: str = "localLatin_runs",
    subfolder: str = "ff1_lata_postact",
) -> None:
    if not local_run_dir.exists():
        raise FileNotFoundError(f"Local run dir not found: {local_run_dir}")

    service = build_drive_service(credentials_path)
    root_id = get_or_create_folder(service, drive_root_folder)
    target_id = get_or_create_folder(service, subfolder, root_id)

    for path in sorted(local_run_dir.rglob("*")):
        if path.is_dir():
            continue
        upload_file(service, path, target_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync run outputs to Google Drive.")
    parser.add_argument("--local_run_dir", required=True, help="Local run directory.")
    parser.add_argument(
        "--credentials",
        default=os.environ.get("DRIVE_SA_KEY_PATH", "/content/sa_drive_key.json"),
        help="Path to service account JSON key.",
    )
    parser.add_argument(
        "--drive_root_folder",
        default="localLatin_runs",
        help="Drive root folder to use/create.",
    )
    parser.add_argument(
        "--subfolder",
        default="ff1_lata_postact",
        help="Subfolder under the Drive root folder.",
    )
    args = parser.parse_args()
    sync_run_dir(
        Path(args.local_run_dir),
        args.credentials,
        drive_root_folder=args.drive_root_folder,
        subfolder=args.subfolder,
    )


if __name__ == "__main__":
    main()
