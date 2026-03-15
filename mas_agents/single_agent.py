"""
Single-Agent System (SAS) for BrowseComp-Plus.

Paper §3.1 / Table 2:
    A = {a1}
    No coordination (baseline)
    Max 10 iterations of search + reasoning

Adapted from plancraft/agents/copilot_single.py.
"""

from copilot import CopilotClient

from .base_agent import CopilotBaseAgent, RetrieverTool
from .llm_client import call_copilot_with_retry
from .prompts import SYSTEM_PROMPT, QUERY_TEMPLATE

MAX_ITERATIONS = 10


class CopilotSingleAgent(CopilotBaseAgent):
    """
    Single Agent using the GitHub Copilot SDK.
    Iteratively searches and reasons until producing a final answer
    or reaching MAX_ITERATIONS.
    """

    def __init__(self, model_name: str, client: CopilotClient, retriever: RetrieverTool):
        super().__init__(model_name, client, retriever)

    async def act(self, context: str | None) -> str:
        # 1. Update history with context (query or search results)
        if context:
            self.conversation.append({"role": "user", "content": context})

        # 2. Iterative search loop
        for iteration in range(MAX_ITERATIONS):
            self.step_count += 1

            # Call Copilot
            action_text = await call_copilot_with_retry(
                self.client,
                self.model_name,
                self.conversation,
                SYSTEM_PROMPT,
            )

            self.conversation.append({"role": "model", "content": action_text})
            self.log(f"Iteration {iteration+1}/{MAX_ITERATIONS}: {action_text[:120]}...")

            # Check if agent wants to search
            search_query = self._extract_search_query(action_text)
            if search_query and not self._is_final_answer(action_text):
                search_results = self._execute_search(search_query)
                self.log(f"  Search: '{search_query}' -> {len(search_results)} chars")
                self.conversation.append({"role": "user", "content": f"Search results for '{search_query}':\n\n{search_results}"})
                continue

            # Final answer reached (or no search action)
            return action_text

        # Max iterations reached — return last response
        self.log(f"Max iterations ({MAX_ITERATIONS}) reached")
        return self.conversation[-1]["content"] if self.conversation else "No answer produced."

    async def run_query(self, query_id: str, query_text: str) -> dict:
        """
        Run a complete query from start to finish.

        Args:
            query_id: The query ID from the dataset
            query_text: The question text

        Returns:
            BrowseComp-Plus formatted output dict
        """
        self.reset(query_id, query_text)

        # Format the initial query
        initial_prompt = QUERY_TEMPLATE.format(question=query_text)

        # Run the agent
        await self.act(initial_prompt)

        return self.get_output()
