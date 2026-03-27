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
    updated_code: str = Field(description="The updated and completely fixed Python code.")
    reasoning: str = Field(description="Brief explanation of what was changed and why.")


structured_coder = coder_model.with_structured_output(CodeFix)

SYSTEM_PROMPT = """You are an expert code debugging assistant.

Your job is to help users fix broken Python code through iterative debugging.

MANDATORY WORKFLOW:
1. When the user provides code, ALWAYS run it first using execute_code to see the actual error output.
2. After seeing the error, use generate_fix to get a targeted fix from the code specialist.
3. Extract ONLY the Python code from the generate_fix response (no markdown, no code fences) and run it with execute_code to verify.
4. If it still fails, repeat steps 2-3. Maximum 3 fix attempts total.
5. If you encounter ModuleNotFoundError, use install_package to install the missing package and re-run the code.
6. After the code works OR after 3 failed attempts, provide a clear explanation to the user.

STRICT RULES:
- ALWAYS execute code before claiming it works or is fixed.
- ALWAYS use generate_fix for code modifications. Do NOT write fixes yourself.
- When calling execute_code, pass ONLY raw Python code. No markdown formatting, no code fence blocks.
- NEVER hardcode outputs or remove functionality to avoid errors.
- NEVER suppress errors with blind try/except blocks.
- Keep explanations clear and accessible to both technical and non-technical users.
- After 3 failed fix attempts, explain what went wrong and ask the user for more information.
- You can chat normally if the user asks questions about code, programming concepts, or follow-ups about previous fixes.

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
    """Execute Python code in a sandboxed Docker environment and return the results.
    Returns stdout, stderr, and exit code.
    IMPORTANT: Pass only raw Python code, no markdown formatting."""
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
    """Generate a targeted code fix using the Codestral code specialist model.
    Args:
        code: The broken Python code that needs fixing.
        errors: The error output from executing the code.
        user_intent: The user's description of what the code should do."""
    try:
        response = structured_coder.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a precise code repair system. "
                        "Fix ONLY the reported errors. "
                        "Preserve original logic and structure. "
                        "Do NOT rewrite the entire code unless absolutely necessary. "
                        "Do NOT hardcode outputs or bypass logic. "
                        "Do NOT suppress errors using try/except unless required. "
                        "Do NOT remove functionality to avoid errors. "
                        "If error is ModuleNotFoundError, leave the import unchanged. "
                        "The final code MUST be complete and directly executable."
                    ),
                },
                {
                    "role": "user",
                    "content": (f"User intent:\n{user_intent}\n\nBroken code:\n{code}\n\nErrors:\n{errors}\n\nProvide the complete fixed code and explain your changes."),
                },
            ]
        )
        return f"FIXED CODE:\n```python\n{response.updated_code}\n```\n\nREASONING:\n{response.reasoning}"
    except Exception as e:
        return f"Code generation failed: {str(e)}"


@tool
def install_package(package_name: str, code: str) -> str:
    """Install a Python package and then execute code in the same sandbox session.
    Use correct pip package names:
    - cv2 -> opencv-python
    - PIL -> Pillow
    - sklearn -> scikit-learn
    - bs4 -> beautifulsoup4
    - yaml -> PyYAML
    Args:
        package_name: The pip package name to install.
        code: The Python code to execute after installation."""
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

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": "test-1"}, "recursion_limit": 30}

    result = chatbot.invoke(
        {"messages": [HumanMessage(content="Fix this code:\n\nfor i in range(10)\n    print(i)")]},
        config=config,
    )

    print(result["messages"][-1].content)