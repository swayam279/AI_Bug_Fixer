# from langgraph.graph import StateGraph


class StateGraphSimulator:
    def __init__(self):
        self.nodes = {}
        self.entry_point = None

    def add_node(self, name, func):
        self.nodes[name] = func

    def set_entry_point(self, name):
        self.entry_point = name

    def invoke(self, initial_state):
        if self.entry_point is None:
            raise ValueError("Entry point not set")
        return self.nodes[self.entry_point](initial_state)


def greet_node(state):
    return {"message": "Hello " + state["name"]}


# Simulate the StateGraph
builder = StateGraphSimulator()
builder.add_node("greet", greet_node)
builder.set_entry_point("greet")

# Run the graph
initial_state = {"name": "Alice"}
result = builder.invoke(initial_state)

print(result["message"])
