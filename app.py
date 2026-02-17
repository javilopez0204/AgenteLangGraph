import streamlit as st
import functools
from typing import Annotated, TypedDict, List, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
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
    """Estado del grafo que mantiene el historial de mensajes."""
    messages: Annotated[List[BaseMessage], add_messages]


# --- NODOS DEL GRAFO ---

def search_node(state: AgentState, llm: ChatGoogleGenerativeAI):
    messages = [SystemMessage(content=SEARCH_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def outliner_node(state: AgentState, llm: ChatGoogleGenerativeAI):
    messages = [SystemMessage(content=OUTLINER_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def writer_node(state: AgentState, llm: ChatGoogleGenerativeAI):
    messages = [SystemMessage(content=WRITER_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def should_search(state: AgentState) -> Literal["tools", "outliner"]:
    """
    FIX: Verificación segura de tool_calls. AIMessage siempre tiene el atributo,
    pero otros tipos de mensaje (HumanMessage, ToolMessage) no lo tienen
    o lo tienen vacío, por lo que usamos getattr con fallback.
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:
        return "tools"
    return "outliner"


# --- SERIALIZACIÓN PARA DEBUG ---

def serialize_event(event: dict) -> dict:
    """
    FIX: Los objetos BaseMessage no son JSON-serializables directamente.
    Convertimos cada mensaje a un dict con type + content para poder
    mostrarlos con st.json() sin errores.
    """
    serialized = {}
    for node_name, state_update in event.items():
        serialized[node_name] = {}
        for key, value in state_update.items():
            if key == "messages" and isinstance(value, list):
                serialized[node_name][key] = [
                    {
                        "type": msg.__class__.__name__,
                        "content": str(msg.content)[:500],  # truncamos para legibilidad
                        "tool_calls": getattr(msg, "tool_calls", []),
                    }
                    for msg in value
                ]
            else:
                serialized[node_name][key] = str(value)
    return serialized


# --- FACTORY DEL GRAFO ---

def build_graph(google_api_key: str, tavily_api_key: str, model_name: str):
    """
    FIX: Eliminamos @st.cache_resource con API keys como argumentos.
    El caché de Streamlit no invalida correctamente cuando cambian strings
    sensibles entre reruns. El grafo se reconstruye solo al pulsar el botón,
    lo cual es aceptable dado que la operación es puntual.
    También eliminamos 'convert_system_message_to_human' (deprecado).
    """
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

    workflow.set_entry_point("search")

    workflow.add_conditional_edges(
        "search",
        should_search,
        {"tools": "tools", "outliner": "outliner"}
    )

    workflow.add_edge("tools", "search")
    workflow.add_edge("outliner", "writer")
    workflow.add_edge("writer", END)

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
        # FIX: Selectbox alineado con los modelos reales disponibles
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

        try:
            inputs = {"messages": [HumanMessage(content=topic)]}

            for event in app_graph.stream(inputs):
                # FIX: Serializamos el evento antes de guardarlo para debug
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

                        if "messages" in state_update:
                            last_msg = state_update["messages"][-1]
                            if isinstance(last_msg, AIMessage):
                                final_content = last_msg.content

            status_box.update(label="✅ Proceso completado", state="complete", expanded=False)

            if final_content:
                st.divider()
                st.subheader("📰 Artículo Generado")
                st.markdown(final_content)
            else:
                st.warning("El flujo terminó pero no se detectó contenido final en el nodo Writer.")

            with st.expander("🔍 Ver traza interna (Debug)"):
                st.json(debug_events)

        except Exception as e:
            status_box.update(label="❌ Error durante la ejecución", state="error")
            st.error(f"Ocurrió un error inesperado: {e}")


if __name__ == "__main__":
    main()