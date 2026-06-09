from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mlflow_tracking_uri: str
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    champion_poll_interval: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()