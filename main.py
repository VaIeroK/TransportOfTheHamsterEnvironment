from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from contr.upload import upload_router
from views.graph import graph_router


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE", "PATCH", "PUT"],
    allow_headers=["Content-Type", "Set-Cookie", "Access-Control-Allow-Headers", "Access-Control-Allow-Origin",
                   "Authorization"],
)

app.include_router(
    router=graph_router,
    prefix='/api/view',
    tags=['Data from yolo']
)

app.include_router(
    router=upload_router,
    prefix='/api/upload',
    tags=['Upload video from User'],
)