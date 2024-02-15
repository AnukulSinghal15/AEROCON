from fastapi import FastAPI
from chatgpt import askQuery
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all origins, replace "*" with your React app's origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.get("/")
def userQuery():
      return {"Welcome"};


@app.get("/ask/")
async def userQuery(query:str=None):
      gptAnswer= await askQuery(query)
      return {"gptResponse": gptAnswer};

