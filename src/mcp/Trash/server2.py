from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class AddTool(BaseModel):
    a: int
    b: int

class MultiplyTool(BaseModel):
    a: int
    b: int

@app.post("/add")
async def add_numbers(tool: AddTool):
    return {"result": tool.a + tool.b}

@app.post("/multiply")
async def multiply_numbers(tool: MultiplyTool):
    return {"result": tool.a * tool.b}

# To run this server: uvicorn server:app --reload