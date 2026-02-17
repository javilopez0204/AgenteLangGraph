import streamlit as st
import functools
from typing import Annotated, TypedDict, List, Literal, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig

# --- CONSTANTES Y CONFIGURACIÓN ---
SEARCH_PROMPT = """You are a research assistant. Your job is to search the web for related news that would be relevant to generate the article described by the user.
NOTE: Do not write the article. Just search the web for related news if needed. If you have enough info, stop searching."""

OUTLINER_PROMPT = """You are an content strategist. Your job is to take as input a list of search results along with users instruction on what article they want to write and generate a structured outline for the article."""

WRITER_PROMPT = """You are a senior journalist. Write an article based on the provided outline.
Format:
# TITLE: <title>
## BODY: <body>
NOTE: Stick strictly to the facts provided in the context/outline."""

# --- DEFINICIÓN DE TIPOS ---
class AgentState(TypedDict):
    """Estado del grafo que mantiene el historial de mensajes."""
    messages: Annotated[List[BaseMessage], add_messages]

# --- NODOS DEL GRAFO (Desacoplados) ---
# Usamos inyección de dependencias para el LLM, permitiendo testabilidad.

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
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "outliner"

# --- FACTORY DEL GRAFO (Con Caching) ---

@st.cache_resource(show_spinner="Inicializando Modelos y Grafo...")
def get_graph_app(google_api_key: str, tavily_api_key: str, model_name: str = "gemini-2.5-flash-lite"):
    """
    Inicializa y compila el grafo. Se cachea para evitar reconstrucción en cada renderizado.
    """
    # 1. Configuración de Herramientas y LLM
    search_tool = TavilySearchResults(max_results=5, tavily_api_key=tavily_api_key)
    tools = [search_tool]
    
    llm = ChatGoogleGenerativeAI(
        model=model_name, 
        temperature=0.5,
        google_api_key=google_api_key,
        convert_system_message_to_human=True
    )
    
    # Vinculamos herramientas para el nodo de búsqueda
    llm_with_tools = llm.bind_tools(tools)
    
    # 2. Binding de Nodos (Partial application para inyectar el LLM específico)
    search_bound = functools.partial(search_node, llm=llm_with_tools)
    outliner_bound = functools.partial(outliner_node, llm=llm)
    writer_bound = functools.partial(writer_node, llm=llm)

    # 3. Construcción del Grafo
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

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        g_key = st.text_input("Google API Key", type="password")
        t_key = st.text_input("Tavily API Key", type="password")
        model_selection = st.selectbox("Modelo", ["gemini-1.5-flash", "gemini-2.0-flash-exp"], index=0)
        
        if not (g_key and t_key):
            st.warning("Introduce tus claves para continuar.")
            st.stop()

    # Área principal
    topic = st.text_input("Tema del artículo:", placeholder="Ej: Avances en computación cuántica 2025")

    if st.button("Generar Artículo", type="primary"):
        if not topic:
            st.warning("Por favor escribe un tema.")
            return

        try:
            # Obtener el grafo (cached)
            app_graph = get_graph_app(g_key, t_key, model_selection)
        except Exception as e:
            st.error(f"Error en configuración: {e}")
            st.stop()

        status_box = st.status("🚀 Ejecutando workflow...", expanded=True)
        final_content = ""
        debug_events = []
        
        try:
            inputs = {"messages": [HumanMessage(content=topic)]}
            
            # Stream de eventos
            for event in app_graph.stream(inputs):
                debug_events.append(event)
                
                for node_name, state_update in event.items():
                    # Feedback UI
                    if node_name == "search":
                        status_box.write("🕵️ **Search:** Buscando información...")
                    elif node_name == "tools":
                        status_box.write("🌐 **Tavily:** Obteniendo datos web...")
                    elif node_name == "outliner":
                        status_box.write("📝 **Outliner:** Creando esquema...")
                    elif node_name == "writer":
                        status_box.write("✍️ **Writer:** Redactando artículo...")
                        
                        # Captura segura del contenido final solo en el nodo writer
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