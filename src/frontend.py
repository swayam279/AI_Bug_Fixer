import re
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend import (
    chatbot,
    checkpoint,
    delete_thread_label,
    load_all_thread_labels,
    retrieve_all_threads,
    save_thread_label,
)

st.set_page_config(page_title="Code Fixer", page_icon="🔧", layout="wide")

st.markdown(
    """
<style>
    [data-testid="stSidebar"] .stButton > button {
        height: 2.5rem;
        padding: 0.25rem 0.5rem;
        font-size: 0.85rem;
    }
    [data-testid="stSidebar"] .stColumns {
        align-items: center;
        gap: 0.25rem;
    }
    [data-testid="stSidebar"] .stTextInput > div > div > input {
        height: 2.5rem;
        font-size: 0.85rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

ALLOWED_EXTENSIONS = ["py", "txt"]


def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["session_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state["uploaded_file_content"] = None
    st.session_state["uploaded_file_name"] = None


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    if state and state.values and "messages" in state.values:
        return state.values["messages"]
    return []


def extract_thread_label(content):
    file_match = re.search(r"$$File:\s*(.+?)$$", content)
    cleaned = re.sub(r"$$File:\s*.+?$$", "", content)
    cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)
    cleaned = cleaned.strip()

    if cleaned:
        label = cleaned[:50]
        if len(cleaned) > 50:
            label += "..."
        return label

    if file_match:
        return f"📄 {file_match.group(1)}"

    return None


def get_thread_label(thread_id):
    if thread_id in st.session_state["thread_labels"]:
        return st.session_state["thread_labels"][thread_id]

    messages = load_conversation(thread_id)
    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content:
            label = extract_thread_label(msg.content)
            if label:
                return label
    return None


def rename_thread(thread_id, new_name):
    new_name = new_name.strip()
    if new_name:
        st.session_state["thread_labels"][thread_id] = new_name
        save_thread_label(thread_id, new_name)


def delete_thread(thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    try:
        checkpoints = list(checkpoint.list(config))
        for _cp in checkpoints:
            checkpoint.conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            )
            checkpoint.conn.execute(
                "DELETE FROM writes WHERE thread_id = ?",
                (thread_id,),
            )
            checkpoint.conn.commit()
            break
    except Exception:
        pass

    delete_thread_label(thread_id)

    if thread_id in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].remove(thread_id)

    if thread_id in st.session_state["thread_labels"]:
        del st.session_state["thread_labels"][thread_id]

    if st.session_state["session_id"] == thread_id:
        reset_chat()


def rebuild_history_from_state(thread_id):
    messages = load_conversation(thread_id)
    history = []
    tool_calls_map = {}

    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append(
                {
                    "role": "user",
                    "content": msg.content,
                }
            )
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_map[tc["id"]] = {
                        "name": tc["name"],
                        "args": tc["args"],
                    }
            elif msg.content:
                history.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                    }
                )
        elif isinstance(msg, ToolMessage):
            tc_info = tool_calls_map.get(msg.tool_call_id, {})
            tool_name = tc_info.get("name", "tool")
            tool_args = tc_info.get("args", {})

            if tool_name == "execute_code":
                label = "⚙️ Executed code in sandbox"
            elif tool_name == "generate_fix":
                label = "🔧 Generated fix"
            elif tool_name == "install_package":
                pkg = tool_args.get("package_name", "unknown")
                label = f"📦 Installed {pkg}"
            else:
                label = f"🔨 {tool_name}"

            history.append(
                {
                    "role": "tool",
                    "label": label,
                    "detail": msg.content,
                    "tool_name": tool_name,
                }
            )

    return history


def render_message_history():
    for entry in st.session_state["message_history"]:
        role = entry["role"]

        if role == "user":
            with st.chat_message("user"):
                st.markdown(entry["content"])
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(entry["content"])
        elif role == "tool":
            with st.chat_message("assistant"):
                with st.expander(entry["label"], expanded=False):
                    if entry.get("tool_name") == "generate_fix":
                        st.markdown(entry["detail"])
                    else:
                        st.code(entry["detail"])


def read_uploaded_file(uploaded_file):
    try:
        content = uploaded_file.read().decode("utf-8")
        return content
    except UnicodeDecodeError:
        return None


def build_message(user_text, file_name, file_content):
    parts = []

    if file_content:
        parts.append(f"[File: {file_name}]\n```python\n{file_content}\n```")

    if user_text:
        parts.append(user_text)
    elif file_content and not user_text:
        parts.append("Fix the errors in this code.")

    return "\n\n".join(parts)


def build_display_message(user_text, file_name, file_content):
    parts = []

    if file_content:
        preview = file_content[:200]
        if len(file_content) > 200:
            preview += "\n..."
        parts.append(f"📄 **{file_name}**\n```python\n{preview}\n```")

    if user_text:
        parts.append(user_text)

    return "\n\n".join(parts)


def render_sidebar_thread(thread_id, label):
    is_active = thread_id == st.session_state["session_id"]
    is_editing = st.session_state.get("editing_thread") == thread_id

    if is_editing:
        col_input, col_save, col_cancel = st.sidebar.columns([4, 1, 1])
        with col_input:
            new_name = st.text_input(
                "Rename",
                value=label,
                key=f"rename_input_{thread_id}",
                label_visibility="collapsed",
            )
        with col_save:
            if st.button(
                "✅",
                key=f"save_rename_{thread_id}",
                use_container_width=True,
            ):
                if new_name.strip():
                    rename_thread(thread_id, new_name)
                st.session_state["editing_thread"] = None
                st.rerun()
        with col_cancel:
            if st.button(
                "✖",
                key=f"cancel_rename_{thread_id}",
                use_container_width=True,
            ):
                st.session_state["editing_thread"] = None
                st.rerun()
    else:
        col_thread, col_edit, col_delete = st.sidebar.columns([4, 1, 1])

        with col_thread:
            display_label = f"💬 {label}" if is_active else label
            if st.button(
                display_label,
                key=f"thread_{thread_id}",
                use_container_width=True,
                disabled=is_active,
            ):
                st.session_state["session_id"] = thread_id
                st.session_state["message_history"] = rebuild_history_from_state(thread_id)
                st.session_state["uploaded_file_content"] = None
                st.session_state["uploaded_file_name"] = None
                st.session_state["confirm_delete"] = None
                st.session_state["editing_thread"] = None
                st.rerun()

        with col_edit:
            if st.button(
                "✏️",
                key=f"edit_{thread_id}",
                use_container_width=True,
            ):
                st.session_state["editing_thread"] = thread_id
                st.session_state["confirm_delete"] = None
                st.rerun()

        with col_delete:
            if st.button(
                "🗑️",
                key=f"delete_{thread_id}",
                use_container_width=True,
            ):
                if st.session_state["confirm_delete"] == thread_id:
                    delete_thread(thread_id)
                    st.session_state["confirm_delete"] = None
                    st.rerun()
                else:
                    st.session_state["confirm_delete"] = thread_id
                    st.session_state["editing_thread"] = None
                    st.rerun()

    if st.session_state.get("confirm_delete") == thread_id and not is_editing:
        st.sidebar.warning("Delete this chat?")
        c1, c2 = st.sidebar.columns(2)
        with c1:
            if st.button(
                "Yes",
                key=f"confirm_yes_{thread_id}",
                use_container_width=True,
            ):
                delete_thread(thread_id)
                st.session_state["confirm_delete"] = None
                st.rerun()
        with c2:
            if st.button(
                "No",
                key=f"confirm_no_{thread_id}",
                use_container_width=True,
            ):
                st.session_state["confirm_delete"] = None
                st.rerun()


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "session_id" not in st.session_state:
    st.session_state["session_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "processing" not in st.session_state:
    st.session_state["processing"] = False

if "uploaded_file_content" not in st.session_state:
    st.session_state["uploaded_file_content"] = None

if "uploaded_file_name" not in st.session_state:
    st.session_state["uploaded_file_name"] = None

if "confirm_delete" not in st.session_state:
    st.session_state["confirm_delete"] = None

if "editing_thread" not in st.session_state:
    st.session_state["editing_thread"] = None

if "thread_labels" not in st.session_state:
    st.session_state["thread_labels"] = load_all_thread_labels()

add_thread(st.session_state["session_id"])

st.sidebar.title("🔧 Code Fixer")
st.sidebar.caption("Fix Python code with AI")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Conversations")

threads_to_display = []
for thread_id in st.session_state["chat_threads"][::-1]:
    label = get_thread_label(thread_id)
    if label:
        threads_to_display.append((thread_id, label))

for thread_id, label in threads_to_display:
    render_sidebar_thread(thread_id, label)

st.title("🔧 Code Fixer")
st.caption("Paste your broken Python code, upload a file, or both.")

uploaded_file = st.file_uploader(
    "Upload a Python or text file",
    type=ALLOWED_EXTENSIONS,
    key="file_uploader",
    label_visibility="collapsed",
)

if uploaded_file is not None:
    file_content = read_uploaded_file(uploaded_file)
    if file_content:
        st.session_state["uploaded_file_content"] = file_content
        st.session_state["uploaded_file_name"] = uploaded_file.name
        st.success(f"📄 **{uploaded_file.name}** loaded ({len(file_content)} chars)")
        with st.expander("Preview uploaded code", expanded=False):
            st.code(file_content, language="python")
    else:
        st.error("Could not read file. Make sure it is a valid text file.")

render_message_history()

user_input = st.chat_input("Paste your code, describe the issue, or just send with an uploaded file...")

if user_input and not st.session_state["processing"]:
    st.session_state["processing"] = True

    file_content = st.session_state.get("uploaded_file_content")
    file_name = st.session_state.get("uploaded_file_name")

    full_message = build_message(user_input, file_name, file_content)
    display_message = build_display_message(user_input, file_name, file_content)

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": display_message,
        }
    )

    with st.chat_message("user"):
        st.markdown(display_message)

    st.session_state["uploaded_file_content"] = None
    st.session_state["uploaded_file_name"] = None

    config = {
        "configurable": {"thread_id": st.session_state["session_id"]},
        "recursion_limit": 30,
    }

    tool_calls_cache = {}

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        tool_container = st.container()
        response_placeholder = st.empty()

        collected_tokens = []

        for msg_chunk, _ in chatbot.stream(
            {"messages": [HumanMessage(content=full_message)]},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(msg_chunk, AIMessage):
                if msg_chunk.tool_calls:
                    for tc in msg_chunk.tool_calls:
                        if tc["id"] and tc["id"] not in tool_calls_cache:
                            tool_calls_cache[tc["id"]] = {
                                "name": tc["name"],
                                "args": tc["args"],
                            }
                            tool_name = tc["name"]
                            if tool_name == "execute_code":
                                status_placeholder.info("⚙️ Running code in sandbox...")
                            elif tool_name == "generate_fix":
                                status_placeholder.info("🔧 Generating fix with Codestral...")
                            elif tool_name == "install_package":
                                pkg = tc["args"].get("package_name", "")
                                status_placeholder.info(f"📦 Installing {pkg}...")

                elif msg_chunk.content:
                    status_placeholder.empty()
                    collected_tokens.append(msg_chunk.content)
                    response_placeholder.markdown("".join(collected_tokens))

            elif isinstance(msg_chunk, ToolMessage):
                status_placeholder.empty()
                tc_info = tool_calls_cache.get(msg_chunk.tool_call_id, {})
                tool_name = tc_info.get("name", "tool")
                tool_args = tc_info.get("args", {})

                if tool_name == "execute_code":
                    label = "⚙️ Executed code in sandbox"
                elif tool_name == "generate_fix":
                    label = "🔧 Generated fix"
                elif tool_name == "install_package":
                    pkg = tool_args.get("package_name", "unknown")
                    label = f"📦 Installed {pkg}"
                else:
                    label = f"🔨 {tool_name}"

                with tool_container:
                    with st.expander(label, expanded=False):
                        if tool_name == "generate_fix":
                            st.markdown(msg_chunk.content)
                        else:
                            st.code(msg_chunk.content)

                st.session_state["message_history"].append(
                    {
                        "role": "tool",
                        "label": label,
                        "detail": msg_chunk.content,
                        "tool_name": tool_name,
                    }
                )

        final_response = "".join(collected_tokens)

        if final_response:
            st.session_state["message_history"].append(
                {
                    "role": "assistant",
                    "content": final_response,
                }
            )

    st.session_state["processing"] = False
    st.rerun()