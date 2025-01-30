# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from copilotkit import CopilotKitRemoteEndpoint, LangGraphAgent
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from agent import the_langraph_graph  # Import the new LangGraph agent

app = FastAPI()


@app.get("/")
async def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}


# Define the request model for API interactions
class Message(BaseModel):
    content: str


# Expose a direct chat API endpoint for testing LangGraph
@app.post("/chat/")
async def chat(message: Message):
    """Receives user input and returns a response from the LangGraph agent."""
    try:
        user_input = {"messages": [{"role": "user", "content": message.content}]}
        response = await the_langraph_graph.ainvoke(user_input)
        assistant_message = response["messages"][-1].content
        return {"response": assistant_message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Initialize the CopilotKit SDK using the new agent
sdk = CopilotKitRemoteEndpoint(
    agents=[
        LangGraphAgent(
            name="ai_assistant",
            description="Conversational AI assistant powered by Claude",
            graph=the_langraph_graph,
        )
    ],
)

# Add the CopilotKit endpoint to FastAPI
add_fastapi_endpoint(app, sdk, "/copilotkit_remote")

# if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)
