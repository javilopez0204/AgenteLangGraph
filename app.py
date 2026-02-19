import streamlit as st
import functools
from typing import Annotated, TypedDict, List, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --- CONSTANTES Y CONFIGURACIÓN ---
SEARCH_PROMPT = """You are a research assistant. Your job is to search the web for related news that would be relevant to generate the article described by the user.
NOTE: Do not write the article. Just search the web for related news if needed. If you have enough info, stop searching."""

OUTLINER_PROMPT = """You are a content strategist. Your job is to take as input a list of search results along with the user's instruction on what article they want to write, and generate a structured outline for the article."""

WRITER_PROMPT = """You are a senior journalist. Write an article based on the provided outline.
Format:
# TITLE: <title>
## BODY: <body>
NOTE: Stick strictly to the facts provided in the context/outline."""

AVAILABLE_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash",
]

# --- DEFINICIÓN DE TIPOS ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    error: str  # FIX: campo para propagar errores críticos al frontend


# --- HELPERS ---

def _has_tool_error(state: AgentState) -> str | None:
    """
    FIX: Detecta si el último ToolMessage contiene un error HTTP (ej. 401 Tavily).
    Devuelve el texto del error o None si todo está bien.
    """
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            content = str(msg.content)
            if "Error" in content or "error" in content or "Unauthorized" in content:
                return content
            break
    return None


# --- NODOS DEL GRAFO ---

def search_node(state: AgentState, llm: ChatGoogleGenerativeAI):
    # FIX: Si ya venimos de un ToolMessage con error, no volvemos a invocar el LLM
    # en bucle; marcamos el error y dejamos que should_search corte el flujo.
    tool_error = _has_tool_error(state)
    if tool_error:
        return {"error": tool_error, "messages": []}

    messages = [SystemMessage(content=SEARCH_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response], "error": ""}


def outliner_node(state: AgentState, llm: ChatGoogleGenerativeAI):
    messages = [SystemMessage(content=OUTLINER_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def writer_node(state: AgentState, llm: ChatGoogleGenerativeAI):
    messages = [SystemMessage(content=WRITER_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def should_search(state: AgentState) -> Literal["tools", "outliner", "end_error"]:
    """
    FIX: Tres posibles rutas:
      - 'tools'      → el LLM quiere hacer una búsqueda (hay tool_calls)
      - 'end_error'  → hubo un error en la herramienta, cortamos el grafo
      - 'outliner'   → búsqueda completada, pasamos a generar el esquema
    """
    # Si search_node detectó un error de herramienta, salimos
    if state.get("error"):
        return "end_error"

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:
        return "tools"
    return "outliner"


def error_node(state: AgentState):
    """Nodo terminal que solo propaga el estado de error sin hacer nada más."""
    return {}


# --- SERIALIZACIÓN PARA DEBUG ---

def serialize_event(event: dict) -> dict:
    serialized = {}
    for node_name, state_update in event.items():
        serialized[node_name] = {}
        for key, value in state_update.items():
            if key == "messages" and isinstance(value, list):
                serialized[node_name][key] = [
                    {
                        "type": msg.__class__.__name__,
                        "content": str(msg.content)[:500],
                        "tool_calls": getattr(msg, "tool_calls", []),
                    }
                    for msg in value
                ]
            else:
                serialized[node_name][key] = str(value)
    return serialized


# --- FACTORY DEL GRAFO ---

def build_graph(google_api_key: str, tavily_api_key: str, model_name: str):
    search_tool = TavilySearchResults(max_results=5, tavily_api_key=tavily_api_key)
    tools = [search_tool]

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.5,
        google_api_key=google_api_key,
    )
    llm_with_tools = llm.bind_tools(tools)

    search_bound = functools.partial(search_node, llm=llm_with_tools)
    outliner_bound = functools.partial(outliner_node, llm=llm)
    writer_bound = functools.partial(writer_node, llm=llm)

    workflow = StateGraph(AgentState)

    workflow.add_node("search", search_bound)
    workflow.add_node("outliner", outliner_bound)
    workflow.add_node("writer", writer_bound)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("end_error", error_node)  # FIX: nodo terminal de error

    workflow.set_entry_point("search")

    workflow.add_conditional_edges(
        "search",
        should_search,
        {"tools": "tools", "outliner": "outliner", "end_error": "end_error"}
    )

    workflow.add_edge("tools", "search")
    workflow.add_edge("outliner", "writer")
    workflow.add_edge("writer", END)
    workflow.add_edge("end_error", END)  # FIX: el nodo de error también termina el grafo

    return workflow.compile()


# --- INTERFAZ DE USUARIO ---

def main():
    st.set_page_config(page_title="Agente Redactor AI", page_icon="🤖", layout="wide")

    st.title("🤖 Agente Redactor de Noticias")
    st.markdown("Arquitectura: **LangGraph** + **Gemini** + **Tavily**")

    with st.sidebar:
        st.header("⚙️ Configuración")
        g_key = st.text_input("Google API Key", type="password")
        t_key = st.text_input("Tavily API Key", type="password")
        model_selection = st.selectbox("Modelo", AVAILABLE_MODELS, index=0)

        if not (g_key and t_key):
            st.warning("Introduce tus claves para continuar.")
            st.stop()

    topic = st.text_input("Tema del artículo:", placeholder="Ej: Avances en computación cuántica 2025")

    if st.button("Generar Artículo", type="primary"):
        if not topic:
            st.warning("Por favor escribe un tema.")
            return

        try:
            app_graph = build_graph(g_key, t_key, model_selection)
        except Exception as e:
            st.error(f"Error al inicializar el grafo: {e}")
            st.stop()

        status_box = st.status("🚀 Ejecutando workflow...", expanded=True)
        final_content = ""
        debug_events = []
        workflow_error = ""

        try:
            inputs = {"messages": [HumanMessage(content=topic)], "error": ""}

            for event in app_graph.stream(inputs):
                debug_events.append(serialize_event(event))

                for node_name, state_update in event.items():
                    if node_name == "search":
                        status_box.write("🕵️ **Search:** Buscando información...")
                    elif node_name == "tools":
                        status_box.write("🌐 **Tavily:** Obteniendo datos web...")
                    elif node_name == "outliner":
                        status_box.write("📝 **Outliner:** Creando esquema...")
                    elif node_name == "writer":
                        status_box.write("✍️ **Writer:** Redactando artículo...")
                        if "messages" in state_update and state_update["messages"]:
                            last_msg = state_update["messages"][-1]
                            if isinstance(last_msg, AIMessage):
                                final_content = last_msg.content
                    elif node_name == "end_error":
                        # FIX: capturamos el error del estado para mostrarlo al usuario
                        workflow_error = state_update.get("error", "Error desconocido en la herramienta de búsqueda.")

            # FIX: mostramos el error de forma clara en lugar de un warning genérico
            if workflow_error:
                status_box.update(label="❌ Error en la búsqueda", state="error", expanded=False)
                if "401" in workflow_error or "Unauthorized" in workflow_error:
                    st.error(
                        "🔑 **Tavily API Key inválida o sin permisos (401 Unauthorized).**\n\n"
                        "Verifica que tu clave de Tavily sea correcta en [tavily.com](https://tavily.com)."
                    )
                else:
                    st.error(f"Error durante la búsqueda web:\n\n`{workflow_error}`")
            elif final_content:
                status_box.update(label="✅ Proceso completado", state="complete", expanded=False)
                st.divider()
                st.subheader("📰 Artículo Generado")
                st.markdown(final_content)
            else:
                status_box.update(label="⚠️ Sin contenido", state="complete", expanded=False)
                st.warning("El flujo terminó pero no se detectó contenido final en el nodo Writer.")

            with st.expander("🔍 Ver traza interna (Debug)"):
                st.json(debug_events)

        except Exception as e:
            status_box.update(label="❌ Error durante la ejecución", state="error")
            st.error(f"Ocurrió un error inesperado: {e}")


if __name__ == "__main__":
    main()