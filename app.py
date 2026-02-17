import streamlit as st
from typing import Annotated, TypedDict, List, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
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

# --- FUNCIONES DE LÓGICA (CACHED) ---

@st.cache_resource(show_spinner="Inicializando agentes...")
def create_graph(google_api_key: str, tavily_api_key: str):
    """
    Crea y compila el grafo. Se cachea para evitar re-compilación en cada interacción.
    Se pasan las API keys explícitamente para evitar conflictos globales.
    """
    
    # 1. Inicialización de herramientas y modelo con claves explícitas
    # Nota: Tavily a veces requiere env var, pero intentamos pasarla si la lib lo permite
    # o usamos un contexto seguro. Para TavilySearchResults, pasamos en kwargs si es soportado,
    # si no, lo manejamos con cuidado. Aquí usaremos el binding directo al LLM.
    
    # Configuración segura del LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", 
        temperature=0.5,
        google_api_key=google_api_key,
        convert_system_message_to_human=True # A veces necesario para Gemini
    )
    
    # Configuración de herramienta (Tavily requiere env var típicamente, 
    # pero podemos inyectar via constructor en versiones recientes o manejarlo externamente)
    # Para efectos prácticos en Streamlit cloud, si la lib fuerza env vars, 
    # aseguramos que la key de tavily se pase al wrapper.
    search_tool = TavilySearchResults(
        max_results=5, 
        tavily_api_key=tavily_api_key
    )
    tools = [search_tool]
    
    # Pre-bind de herramientas para eficiencia
    llm_with_tools = llm.bind_tools(tools)

    # 2. Definición de Nodos
    def search_node(state: AgentState):
        """Agente investigador que decide si usar herramientas."""
        messages = [SystemMessage(content=SEARCH_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def outliner_node(state: AgentState):
        """Agente planificador."""
        messages = [SystemMessage(content=OUTLINER_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def writer_node(state: AgentState):
        """Agente redactor."""
        messages = [SystemMessage(content=WRITER_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def should_search(state: AgentState) -> Literal["tools", "outliner"]:
        """Decisión condicional."""
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
    st.markdown("Arquitectura: **LangGraph** + **Gemini 1.5** + **Tavily**")

    # Sidebar
    with st.sidebar:
        st.header("🔐 Credenciales")
        g_key = st.text_input("Google API Key", type="password")
        t_key = st.text_input("Tavily API Key", type="password")
        
        if not (g_key and t_key):
            st.warning("Introduce tus claves para continuar.")
            return

    # Instanciación del grafo (Cacheada)
    try:
        app_graph = create_graph(g_key, t_key)
    except Exception as e:
        st.error(f"Error al inicializar el grafo: {e}")
        return

    # Área principal
    topic = st.text_input("Tema del artículo:", placeholder="Ej: Impacto de la IA en la medicina 2025")

    if st.button("Generar Artículo", type="primary", disabled=not topic):
        status_box = st.status("🚀 Ejecutando workflow...", expanded=True)
        
        try:
            inputs = {"messages": [HumanMessage(content=topic)]}
            final_messages = []
            
            # Iterar sobre los eventos del grafo
            for event in app_graph.stream(inputs):
                for node_name, state_update in event.items():
                    # Feedback visual según el nodo activo
                    if node_name == "search":
                        status_box.write("🕵️ **Investigador:** Analizando información...")
                    elif node_name == "tools":
                        status_box.write("🌐 **Herramienta:** Buscando en la web...")
                    elif node_name == "outliner":
                        status_box.write("📝 **Planificador:** Creando esquema...")
                    elif node_name == "writer":
                        status_box.write("✍️ **Redactor:** Escribiendo artículo final...")
                    
                    # Actualizamos la referencia de los mensajes
                    if "messages" in state_update:
                        final_messages = state_update["messages"]

            status_box.update(label="✅ ¡Completado!", state="complete", expanded=False)

            # Mostrar resultado final
            if final_messages:
                # Obtenemos el último mensaje del estado final
                # Nota: En stream, state_update es parcial, pero al final del loop
                # necesitamos el último mensaje generado por el nodo writer.
                # Una forma segura es invocar el grafo y guardar el output final,
                # pero stream permite UX. Asumimos que el último 'writer' output es el final.
                
                st.divider()
                st.subheader("📰 Resultado Final")
                # El mensaje final debería ser del writer.
                # Si usamos .stream(), el último yield contiene el delta.
                # Para mostrar el texto completo, inspeccionamos el último mensaje recibido.
                last_msg = final_messages[-1]
                st.markdown(last_msg.content)
                
                with st.expander("Ver proceso detallado (Debug)"):
                    st.write(event) # Muestra el último estado crudo

        except Exception as e:
            status_box.update(label="❌ Error", state="error")
            st.error(f"Ocurrió un error en la ejecución: {e}")

if __name__ == "__main__":
    main()