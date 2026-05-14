from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ocr_lang: str = "ru"
    ocr_device: str = "gpu:0"
    max_upload_mb: int = 50
    pdf_dpi: int = 200
    max_pdf_pages: int = 100
    paddle_pdx_model_source: str = "BOS"

    text_detection_model: str = "PP-OCRv5_mobile_det"
    text_recognition_model: str = "eslav_PP-OCRv5_mobile_rec"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
