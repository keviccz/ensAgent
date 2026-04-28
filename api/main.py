from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import config, chat, pipeline, data, annotation, agents

app = FastAPI(title="EnsAgent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(config.router)
app.include_router(chat.router)
app.include_router(pipeline.router)
app.include_router(data.router)
app.include_router(annotation.router)
app.include_router(agents.router)

@app.get("/health")
def health():
    return {"status": "ok"}
