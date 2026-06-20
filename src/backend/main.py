from claude_dj.observability import init_sentry

init_sentry()

from fastapi import FastAPI


app = FastAPI(title="ClaudeDJ Backend", version="0.1.0")


@app.get("/")
def read_root() -> dict[str, str]:
    return {"service": "claude-dj-backend", "status": "ok"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
