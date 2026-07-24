"""
Agent hierarchy for MicroWorld: institutional, retail, and AI-assisted retail.

L3 agents act within a firm's risk budget (MFC internal problem).
L2 agents compete across firms (MFG between firms).
L1 is handled by the top-level MFG module (game.py).
"""
from .institutional import InstitutionalAgent, AgentType
from .retail_ai import RetailAIAgent, InvestorType

__all__ = [
    "InstitutionalAgent",
    "AgentType",
    "RetailAIAgent",
    "InvestorType",
]
