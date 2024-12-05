from typing import Optional


# Custom API error codes
INTERNAL_ERROR = 6200
DATASET_NOT_FOUND = 6500
UPLOAD_IN_PROGRESS = 6505
MODEL_NOT_FOUND = 6601
UPGRADE_PLAN = 6754


class SulieError(Exception):
    
    def __init__(self, code: int, detail: Optional[str]):
        self.code = code
        self.detail = detail

    def __repr__(self) -> str:
        return f"{self.detail} {self.code}"