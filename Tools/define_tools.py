from langchain.tools import tool


@tool(parse_docstring=True)
def search_orders(
    user_id: str,
    status: str,
    limit: int
) -> str:
    """Search for user orders by status
    Use this when the user asks about order history or want to check order status. Always filter by the provided status.
    
    Args:
        user_id: unique identifier for the user
        status: order status: 'pending', 'shipped', or 'delivered'
        limit: maximum number of results to return
    """
    # Implementation here...
    pass
