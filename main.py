from fastapi import FastAPI, Form, Request, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pymongo import MongoClient
from bson import ObjectId

import os
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
import hashlib
from pydantic import BaseModel

load_dotenv()
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", "mysecret"))

MONGO_DETAILS = os.getenv("MONGO_DETAILS", "None")
solar_key = os.getenv('UPSTAGE_API_KEY')
print(MONGO_DETAILS)

client = MongoClient(MONGO_DETAILS)

#database = client.omoknuni
#collection = database.chatCluster

database = client.get_database("omoknuni")
collection = database.get_collection("chatCluster")
dialogue = database.get_collection("dialogue")

print("connect success")


# 정적 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def hash_username(username: str) -> str:
    return hashlib.sha256(username.encode()).hexdigest()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse)
async def read_sign(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/signup")
async def signup(request: Request, name: str = Form(...), id: str = Form(...)):
    user = collection.find_one({"id": hash_username(id)})
    if user is None :
        document = {"name": name, "id": hash_username(id)}
        collection.insert_one(document)
        return templates.TemplateResponse("signup.html", {"request": request, "name": name, "id":id})
    else :
        return templates.TemplateResponse("signup.html", {"request": request, "name": "이미 존재하는 아이디입니다", "id":"아이디를 수정해서 다시 시도하세요"})

@app.post("/login")
async def login(request: Request, name: str = Form(...), id: str = Form(...)):
    user = collection.find_one({"id": hash_username(id)})
    if user is not None :
        if (hash_username(id) == user.get("id")) and (name == user.get("name")):
            #return await main(request, id)
            request.session["id"] = id
            return RedirectResponse(url=f"/main", status_code=303)
            #return templates.TemplateResponse("main.html", {"request": request, "name": name, "id": id})
    return templates.TemplateResponse("login.html", {"request": request, "name": "이름 또는 아이디가 맞지 않습니다", "id": "다시 시도하세요"})

@app.get("/main", response_class=HTMLResponse)
async def main(request: Request):
    id = request.session.get("id")
    if id is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    dialogues = dialogue.find({"id": hash_username(id)})
    dialogues_list = list(dialogues)
    if dialogues_list != []:
        dialogues_Q = [[str(dialogue["Q"]), str(dialogue["A"])] for dialogue in dialogues_list]
        print(dialogues_list)
        return templates.TemplateResponse("main.html", {"request": request, "id": id, "dialogue":dialogues_Q})
    else :
        return templates.TemplateResponse("main.html", {"request": request, "id": id, "dialogue": ["No dialogue"]})

@app.get("/id")
async def get_id(request: Request):
    id = request.session.get("id")
    if id is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"id": id}

@app.get("/chat")
async def chat(request: Request):
    id = request.session.get("id")
    return templates.TemplateResponse("client.html", {"request": request, "id":id})

class MessageData(BaseModel):
    question: str
    answer: str


@app.post("/message")
async def save_message(request:Request, message: MessageData):
    print(f"Received message: {message.question}")
    try :
        id = request.session.get("id")
        if id is None:
            raise HTTPException(status_code=401, detail="Not authenticated")
        hashed_id = hash_username(id)
        found_message = dialogue.find_one({"id": hashed_id, "Q": message.question, "A": message.answer})

        if not found_message:
            document = {"id": hashed_id, "Q": message.question, "A": message.answer}
            dialogue.insert_one(document)
    except Exception as e:
        print(f"Error: {e}")

    return {"status": "success", "message": message.dict()}

@app.get("/view/{_id}", response_class=HTMLResponse)
def view(request:Request, _id : str):
    try:
        id = request.session.get("id")
        object_id = ObjectId(_id)
        dial = dialogue.find_one({"_id": object_id})
        if not dial:
            raise HTTPException(status_code=404, detail="Document not found")
        question, answer = dial.get("Q"), dial.get("A")
        return templates.TemplateResponse("view.html", {"request": request, "question": question, "answer": answer, "_id": _id, "id":id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/edit/{_id}", response_class=HTMLResponse)
async def edit(request: Request, _id : str):
    try:
        id = request.session.get("id")
        object_id = ObjectId(_id)
        dial = dialogue.find_one({"_id": object_id})
        if not dial:
            raise HTTPException(status_code=404, detail="Document not found")
        question, answer = dial.get("Q"), dial.get("A")
        return templates.TemplateResponse("edit.html", {"request": request, "question": question, "answer": answer, "_id": _id, "id":id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dialid", response_class=JSONResponse)
async def id_message(request:Request, message: MessageData):
    try:
        dial = dialogue.find_one({"Q": message.question, "A": message.answer})
        _id = dial.get("_id")
        print(str(_id))
        return {"status": "success", "_id": str(_id)}
        #return RedirectResponse(url=f"/editt/{_id}", status_code=303)
        #
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delmessage", response_class=JSONResponse)
async def del_message(request:Request):
    try:
        body = await request.body()
        _id = body.decode('utf-8')
        print(_id)
        object_id = ObjectId(_id)
        #dial = dialogue.find_one({"_id":object_id})
        result = dialogue.delete_one({"_id": object_id})

        if result.deleted_count == 1:
            return {"status": "delete success", "_id": str(_id)}
        else:
            return {"status": "no dialogue", "_id": str(_id)}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class editmessage(BaseModel):
    idedit : str
    updatedQuestion : str
    updatedAnswer: str

@app.post("/editmessage", response_class=JSONResponse)
async def edit_QA(request:Request, edit:editmessage):
    try:
        id = request.session.get("id")
        print(edit)
        _id = edit.idedit
        object_id = ObjectId(_id)
        dial = dialogue.find_one({"_id": object_id})
        if edit.updatedQuestion == "" :
            edit.updatedQuestion = dial.get("Q")[len(id)+1:]
        if edit.updatedAnswer == "":
            edit.updatedAnswer = dial.get("A")

        if edit.updatedQuestion == "" and edit.updatedAnswer == "":
            return {"status": "edit fail", "_id": str(_id)}
        else :
            update_result = dialogue.update_one(
                {"_id": object_id},
                {"$set": {"Q": edit.updatedQuestion, "A": edit.updatedAnswer}}
            )

        if update_result.modified_count == 1:
            return {"status": "edit success", "_id": str(_id)}
        else:
            return {"status": "no dialogue or no update made", "_id": str(_id)}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=UpstageEmbeddings(model="solar-embedding-1-large"),
)
retriever = vectorstore.as_retriever()

llm = ChatUpstage(streaming=True)

chat_with_history_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an helpful senior barista for advising tasks for your jounior baristas. You are an expert of making coffees, servicing customers,and managing cafe.
The recipe or work guideline is given in your context, you must advise to he or she what to do carefully. Please be patient and listen to all chats from your jounior barista.
You have to tell each step by step process of making coffee or other tasks. And also, you must provide considerate explanation for each step and question carefully.
Your opponent requires more strict and detail explain than others. And most important role of you is check whether he or she works with the proper procedures of Cafe Job Description as below. When 
If they ask about their current duties, refer to this Cafe Job Description to provide your response.
Cafe Job Description
*Opening Shift Part-Timer
#Upon Arrival
1. Turn on all store switches
2. Open the back door
3. Turn on the refrigerator
4. Organize the supplies received from headquarters
#During Work Hours
1. Refer to the recipe on the left wall to make beverages
2. Organize cookies and cakes during downtime
3. Check the expiration date of coffee beans and replace them as needed
4. Do the dishes
#Before Leaving
1. Record work hours on the work schedule
If you don't know the answer, just say that you don't know.
---
CONTEXT:
{context}
         """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}"),
    ]
)

chain = chat_with_history_prompt | llm | StrOutputParser()

class CafeManualSchema(BaseModel):
    query: str

@tool
def cafe_manual(query: str) -> str:
    """This is for query for barista tasks in cafe. So, any queries as coffe recipes or customer services should be answered by this.
    Query for coffee, recipes, barista chores, and anything possible in daily cafe works.
    """
    search_result = retriever.invoke(query)
    return search_result


tools = [cafe_manual]
llm_with_tools = llm.bind_tools(tools)


async def call_tool_func(tool_call):
    tool_name = tool_call["name"].lower()
    if tool_name not in globals():
        print("Tool not found", tool_name)
        return None
    selected_tool = globals()[tool_name]
    return selected_tool.invoke(tool_call["args"])

chat_history_human = {}
chat_history_ai = {}

async def chat_llm(message, history = []):
    history_langchain_format = []
    chat_history_human[len(chat_history_human)] = message
    # search_result = retriever.invoke(message)
    # print(search_result)
    if history == []:
        latest_history = []
    elif len(history) == 0:
        latest_history = []
    else:
        latest_history = [history[-1][0], history[-1][1]]

    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
        # if human not in chat_history_human.values():
        # chat_history_ai[len(chat_history_ai)] = ai

    for _ in range(2):  # try 3 times
        tool_calls = llm_with_tools.invoke(message).tool_calls
        if tool_calls:
            break
        else:
            pass
            # print("try again")

    context = ""
    for tool_call in tool_calls:
        context += str(await call_tool_func(tool_call))

    # search_result.append(context)
    #generator = chain.stream({"message": message, "context": context, "history": latest_history})
    generator = chain.invoke({"message": message, "context": context, "history": latest_history})

    assistant = ""
    for gen in generator:
        assistant += gen
        yield assistant
    chat_history_ai[len(chat_history_ai)] = assistant


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            response_text = ""
            async for response in chat_llm(data):
                response_text = response
            await websocket.send_text(response_text)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)