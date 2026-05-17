from pydantic import BaseModel, Field


class CustomerSupportTicket(BaseModel):
    """Structured ticket information extracted from customer messages."""
    category: str = Field(
        description="Issue category: 'billing', 'technical', 'account', or 'product'"
    )
    priority: str = Field(
        description="Urgency level: 'low', 'medium', 'high', or 'critical'"
    )
    summary: str = Field(
        description="One sentence summary of the customer's issue"
    )
    customer_sentiment: str = Field(
        description="Customers emotional tone: 'frustrated', 'satisfied', or 'neutral'"
    )
