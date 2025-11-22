"""Tools for external knowledge and computation"""

from .tool_controller import ToolUseController
from .web_search import WebSearchTool
from .rag_retrieval import RAGRetrieval


def create_tool_env(serpapi_key=None):
    """
    Create a complete tool environment for TARS.
    
    Returns:
        dict of callable tools
    """
    
    # Initialize tools
    web_search = WebSearchTool(api_key=serpapi_key)
    rag = RAGRetrieval()
    
    # Calculator tool
    def calculator(expression):
        try:
            # Safe eval for basic math
            if isinstance(expression, str):
                result = eval(expression, {"__builtins__": {}}, {
                    'abs': abs, 'min': min, 'max': max,
                    'sum': sum, 'pow': pow, 'round': round
                })
            else:
                result = expression
            return f"Result: {result}"
        except Exception as e:
            return f"Calculator error: {e}"
    
    # Code executor (simplified - use proper sandbox in production)
    def code_executor(code):
        try:
            exec_globals = {}
            exec(str(code), exec_globals)
            return "Code executed successfully"
        except Exception as e:
            return f"Code execution error: {e}"
    
    return {
        'web_search': web_search,
        'rag_retrieve': rag,
        'calculate': calculator,
        'execute_code': code_executor,
        'reason': lambda x: "Internal reasoning completed"
    }


__all__ = [
    'ToolUseController',
    'WebSearchTool',
    'RAGRetrieval',
    'create_tool_env'
]