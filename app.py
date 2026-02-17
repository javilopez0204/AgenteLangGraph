import streamlit as st
from typing import Annotated, TypedDict, List, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --- CONSTANTES Y PROMPTS ---
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
    messages: Annotated[List[BaseMessage], add_messages]

# --- FUNCIONES DE LÓGICA ---

# ⚠️ CORRECCIÓN: Eliminamos @st.cache_resource para evitar errores de conexión/sesión cerrada en segundas ejecuciones.
def create_graph(google_api_key: str, tavily_api_key: str):
    """
    Crea y compila el grafo. Se genera fresco en cada ejecución para garantizar conexiones activas.
    """
    
    # 1. Inicialización de herramientas y modelo
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", # Usamos 1.5 Flash que es muy estable y rápido
        temperature=0.5,
        google_api_key=google_api_key,
        convert_system_message_to_human=True
    )
    
    search_tool = TavilySearchResults(
        max_results=5, 
        tavily_api_key=tavily_api_key
    )
    tools = [search_tool]
    
    llm_with_tools = llm.bind_tools(tools)

    # 2. Definición de Nodos
    def search_node(state: AgentState):
        messages = [SystemMessage(content=SEARCH_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def outliner_node(state: AgentState):
        messages = [SystemMessage(content=OUTLINER_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def writer_node(state: AgentState):
        messages = [SystemMessage(content=WRITER_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def should_search(state: AgentState) -> Literal["tools", "outliner"]:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return "outliner"

    # 3. Construcción del Grafo
    workflow = StateGraph(AgentState)

    workflow.add_node("search", search_node)
    workflow.add_node("outliner", outliner_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("search")

    workflow.add_conditional_edges(
        "search",
        should_search,
        {
            "tools": "tools",
            "outliner": "outliner"
        }
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
        st.header("🔐 Credenciales")
        g_key = st.text_input("Google API Key", type="password")
        t_key = st.text_input("Tavily API Key", type="password")
        
        if not (g_key and t_key):
            st.warning("Introduce tus claves para continuar.")
            st.stop() # Detiene la ejecución limpiamente si no hay claves

    # Área principal
    topic = st.text_input("Tema del artículo:", placeholder="Ej: Impacto de la IA en la medicina 2025")

    if st.button("Generar Artículo", type="primary"):
        if not topic:
            st.warning("Por favor escribe un tema.")
        else:
            # Inicializamos el grafo fresco aquí
            try:
                app_graph = create_graph(g_key, t_key)
            except Exception as e:
                st.error(f"Error al inicializar el modelo: {e}")
                st.stop()

            status_box = st.status("🚀 Ejecutando workflow...", expanded=True)
            
            final_article_content = None # Variable para guardar el resultado final
            
            try:
                inputs = {"messages": [HumanMessage(content=topic)]}
                
                # Iterar sobre los eventos del grafo
                for event in app_graph.stream(inputs):
                    for node_name, state_update in event.items():
                        
                        # Feedback visual
                        if node_name == "search":
                            status_box.write("🕵️ **Investigador:** Analizando información...")
                        elif node_name == "tools":
                            status_box.write("🌐 **Herramienta:** Buscando en la web...")
                        elif node_name == "outliner":
                            status_box.write("📝 **Planificador:** Creando esquema...")
                        elif node_name == "writer":
                            status_box.write("✍️ **Redactor:** Escribiendo artículo final...")
                            # ⚠️ CAPTURA SEGURA: Guardamos explícitamente lo que produce el writer
                            if "messages" in state_update:
                                final_article_content = state_update["messages"][-1].content
                        
                status_box.update(label="✅ ¡Completado!", state="complete", expanded=False)

                # Mostrar resultado final fuera del bucle
                if final_article_content:
                    st.divider()
                    st.subheader("📰 Resultado Final")
                    st.markdown(final_article_content)
                else:
                    st.error("El flujo terminó pero no se detectó contenido final del redactor.")

            except Exception as e:
                status_box.update(label="❌ Error", state="error")
                st.error(f"Ocurrió un error en la ejecución: {e}")

if __name__ == "__main__":
    main()