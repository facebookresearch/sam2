class DownloadError(Exception):
    def __init__(self, message):
        self.message = message


class DeleteError(Exception):
    def __init__(self, message):
        self.message = message


class UploadError(Exception):
    def __init__(self, message):
        self.message = message
