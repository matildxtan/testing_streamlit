from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import pickle
import os

SCOPES = [
    "https://www.googleapis.com/auth/drive.file"
]


def authenticate():

    creds = None

    if os.path.exists("token.pickle"):

        with open(
            "token.pickle",
            "rb"
        ) as token:

            creds = pickle.load(
                token
            )

    if not creds or not creds.valid:

        if (
            creds and
            creds.expired and
            creds.refresh_token
        ):

            creds.refresh(
                Request()
            )

        else:

            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json",
                SCOPES
            )

            creds = flow.run_local_server(
                port=0
            )

        with open(
            "token.pickle",
            "wb"
        ) as token:

            pickle.dump(
                creds,
                token
            )

    service = build(
        "drive",
        "v3",
        credentials=creds
    )

    return service


def upload_file(filepath, folder_id):

    service = authenticate()

    file_metadata = {
        "name": os.path.basename(filepath),
        "parents": [folder_id]
    }

    media = MediaFileUpload(
        filepath,
        resumable=True
    )

    service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id"
    ).execute()

    print(
        "Uploaded:",
        filepath
    )