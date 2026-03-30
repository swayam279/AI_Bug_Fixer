import sqlite3
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from llm_sandbox import SandboxBackend, SandboxSession
from pydantic import BaseModel, Field

load_dotenv()

agent_model = ChatMistralAI(model="mistral-large-latest")
coder_model = ChatMistralAI(model="codestral-latest")


class CodeFix(BaseModel):
    updated_code: str = Field(description="The complete fixed Python code.")
    reasoning: str = Field(description="Brief explanation of what was changed and why.")


structured_coder = coder_model.with_structured_output(CodeFix)

SYSTEM_PROMPT = """You are an expert code debugging assistant.

Your job is to help users fix broken Python code through iterative debugging.

ARCHITECTURE AWARENESS:
- You operate in a system where code is executed in a sandboxed Docker environment on the SERVER side.
- The user does NOT interact with Docker. They have their own separate Python environment.
- If execute_code returns an error containing "Docker", "CreateFile", "server API version", "connection refused", or similar infrastructure errors, this is a SERVER-SIDE issue, NOT the user's problem.
- In such cases:
  - Do NOT tell the user to install or start Docker.
  - Do NOT assume the user's Python is broken.
  - Simply inform the user: "The code execution environment is temporarily unavailable. I'll analyze your code statically instead."
  - Then analyze the code based on your own knowledge without execution, and provide your best fix.
  - Still use generate_fix to produce the fix.
  - Clearly state that the fix has NOT been verified by execution due to the environment issue.
- NEVER expose internal infrastructure details (Docker, sandbox, containers) to the user.
- From the user's perspective, you simply "run their code" — how you do it is irrelevant to them.

SCOPE OF CONVERSATION:
- You ONLY discuss topics related to code, programming, debugging, and software development.
- You may exchange basic greetings (hello, hi, thanks, goodbye, etc.).
- You may answer questions about programming concepts, languages, libraries, errors, and best practices.
- You may discuss previous fixes, code improvements, and follow-up questions about code you have worked on.
- You MUST NOT answer questions unrelated to code or programming. This includes but is not limited to:
  - General knowledge (history, geography, science, politics, sports, etc.)
  - Personal opinions or advice
  - Math problems that are not part of a coding context
  - Creative writing, stories, or jokes unrelated to programming
- If a user asks an off-topic question, respond with:
  "I'm a code debugging assistant and can only help with programming-related questions. Feel free to share any code you'd like me to fix or ask me anything about coding!"
- Do NOT apologize excessively. Just redirect politely and concisely.

MANDATORY WORKFLOW:
1. When the user provides code, ALWAYS attempt to run it first using execute_code to see the actual error output.
2. If execute_code fails due to an infrastructure error (Docker/sandbox issue), skip execution and analyze the code statically.
3. After seeing the error (or analyzing statically), use generate_fix to get a targeted fix from the code specialist.
4. Extract ONLY the Python code from the generate_fix response (no markdown, no code fences) and run it with execute_code to verify.
5. If execution is unavailable, present the fix without verification and clearly state it was not tested.
6. If it still fails, repeat steps 3-4. Maximum 3 fix attempts total.
7. If you encounter ModuleNotFoundError, use install_package to install the missing package and re-run the code.
8. After the code works OR after 3 failed attempts, provide a clear explanation to the user.

STRICT RULES:
- ALWAYS attempt to execute code before claiming it works or is fixed.
- ALWAYS use generate_fix for code modifications. Do NOT write fixes yourself.
- When calling execute_code, pass ONLY raw Python code. No markdown formatting, no code fence blocks.
- NEVER hardcode outputs or remove functionality to avoid errors.
- NEVER suppress errors with blind try/except blocks.
- NEVER mention Docker, containers, sandboxes, or server infrastructure to the user.
- NEVER tell the user to fix the execution environment. That is not their responsibility.
- Keep explanations clear and accessible to both technical and non-technical users.
- After 3 failed fix attempts, explain what went wrong and ask the user for more information.

INPUT FORMAT:
- Users may provide code directly in their message or upload .py/.txt files.
- When a file is uploaded, the code will be provided with a label like [File: filename.py].
- Treat uploaded file code exactly the same as pasted code.
- If both a message and file are provided, the message may contain context about the file.

WHEN EXPLAINING FIXES:
- Tell the user what was wrong in simple terms.
- Explain what was changed.
- Show the final working code exactly once in a single code block.
- If the user is non-technical, avoid jargon.
- Do NOT repeat yourself. State each point once.
- Keep the response concise and well-structured.
- Use this format:

**Problem:** [one sentence]
**Fix:** [one sentence]
**Working Code:**
```python
[code here]
Result: [one sentence about the output]"""


@tool
def execute_code(code: str) -> str:
    """Execute Python code in a sandboxed Docker environment and return the results."""
    try:
        session = SandboxSession(
            backend=SandboxBackend.DOCKER,
            lang="python",
            keep_template=True,
        )
        session.open()
        result = session.run(code)
        session.close()

        stdout = result.stdout if result.stdout else ""
        stderr = result.stderr if result.stderr else ""

        if len(stdout) > 3000:
            stdout = stdout[:3000] + "\n... [OUTPUT TRUNCATED]"
        if len(stderr) > 3000:
            stderr = stderr[:3000] + "\n... [OUTPUT TRUNCATED]"

        return f"Exit Code: {result.exit_code}\nStdout:\n{stdout}\nStderr:\n{stderr}"
    except Exception as e:
        return f"Execution environment error: {str(e)}"


@tool
def generate_fix(code: str, errors: str, user_intent: str) -> str:
    """Generate a targeted code fix using the Codestral model."""
    try:
        response = structured_coder.invoke(
            [
                {
                    "role": "system",
                    "content": "You are a precise code repair system.",
                },
                {
                    "role": "user",
                    "content": (f"User intent:\n{user_intent}\n\nBroken code:\n{code}\n\nErrors:\n{errors}"),
                },
            ]
        )

        return f"FIXED CODE:\npython\n{response.updated_code}\n\n\nREASONING:\n{response.reasoning}"
    except Exception as e:
        return f"Code generation failed: {str(e)}"


@tool
def install_package(package_name: str, code: str) -> str:
    """Install a Python package and execute code."""
    try:
        session = SandboxSession(
            backend=SandboxBackend.DOCKER,
            lang="python",
            keep_template=True,
        )
        session.open()

        install_code = f"import subprocess; subprocess.run(['pip', 'install', '-q', '{package_name}'], capture_output=True, text=True)"
        session.run(install_code)

        result = session.run(code)
        session.close()

        stdout = result.stdout if result.stdout else ""
        stderr = result.stderr if result.stderr else ""

        if len(stdout) > 3000:
            stdout = stdout[:3000] + "\n... [OUTPUT TRUNCATED]"
        if len(stderr) > 3000:
            stderr = stderr[:3000] + "\n... [OUTPUT TRUNCATED]"

        return f"Package '{package_name}' installed.\nExit Code: {result.exit_code}\nStdout:\n{stdout}\nStderr:\n{stderr}"
    except Exception as e:
        return f"Installation/execution failed: {str(e)}"


tools = [execute_code, generate_fix, install_package]
model_with_tools = agent_model.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def agent_node(state: ChatState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)

conn = sqlite3.connect(database="code_fixer.db", check_same_thread=False)
checkpoint = SqliteSaver(conn=conn)

conn.execute("""
CREATE TABLE IF NOT EXISTS thread_labels (
    thread_id TEXT PRIMARY KEY,
    label TEXT NOT NULL
)
""")
conn.commit()


def save_thread_label(thread_id, label):
    conn.execute(
        "INSERT OR REPLACE INTO thread_labels (thread_id, label) VALUES (?, ?)",
        (thread_id, label),
    )
    conn.commit()


def load_thread_label(thread_id):
    cursor = conn.execute(
        "SELECT label FROM thread_labels WHERE thread_id = ?",
        (thread_id,),
    )
    row = cursor.fetchone()
    if row:
        return row[0]
    return None


def load_all_thread_labels():
    cursor = conn.execute("SELECT thread_id, label FROM thread_labels")
    return {row[0]: row[1] for row in cursor.fetchall()}


def delete_thread_label(thread_id):
    conn.execute(
        "DELETE FROM thread_labels WHERE thread_id = ?",
        (thread_id,),
    )
    conn.commit()


graph = StateGraph(ChatState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

chatbot = graph.compile(checkpointer=checkpoint)


def retrieve_all_threads():
    all_threads = set()
    for cpt in checkpoint.list(None):
        all_threads.add(cpt.config["configurable"]["thread_id"])
    return list(all_threads)
