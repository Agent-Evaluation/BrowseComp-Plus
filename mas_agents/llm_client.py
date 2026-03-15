"""
Copilot SDK wrapper for BrowseComp-Plus agents.

Uses the GitHub Copilot SDK (copilot) to call LLMs through the
Copilot CLI. Requires the Copilot CLI to be installed and available
in $PATH.

Adapted from plancraft/agents/copilot_llm.py — same retry logic,
prompt building adapted for deep research instead of Minecraft crafting.
"""

import asyncio
from copilot import CopilotClient, SessionConfig, MessageOptions

MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 5
INTER_REQUEST_DELAY = 2.0
DEFAULT_MODEL = "gpt-4.1"


async def get_copilot_client() -> CopilotClient:
    """Create and start a CopilotClient."""
    client = CopilotClient()
    await client.start()
    return client


async def call_copilot_with_retry(
    client: CopilotClient,
    model_name: str,
    messages: list[dict],
    system_prompt: str,
    temperature: float = 0.0,
) -> str:
    """
    Calls a model via the Copilot SDK with rate limiting and retry logic.

    Args:
        client: An already-started CopilotClient.
        model_name: Model to use (e.g. "gpt-5-mini").
        messages: Conversation history as list of {"role": ..., "content": ...} dicts.
        system_prompt: System-level instruction prepended to the conversation.
        temperature: Sampling temperature (0.0 = deterministic).

    Returns:
        The model's response text.
    """
    # Rate limiting
    await asyncio.sleep(INTER_REQUEST_DELAY)

    # Build the full prompt from system prompt + conversation history
    full_prompt = _build_prompt(system_prompt, messages)

    retries = 0
    delay = INITIAL_RETRY_DELAY
    last_error = None

    while retries < MAX_RETRIES:
        session = None
        try:
            session = await client.create_session(
                SessionConfig(model=model_name)
            )

            response = await session.send_and_wait(
                MessageOptions(prompt=full_prompt),
                timeout=120.0,
            )

            if response and response.data and response.data.content:
                return response.data.content.strip()
            else:
                raise Exception("Empty response from Copilot SDK")

        except TimeoutError:
            print(f"  [timeout] Copilot request timed out. Retrying in {delay}s... (attempt {retries+1}/{MAX_RETRIES})")
            last_error = TimeoutError("Request timed out")
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                print(f"  [rate-limit] Retrying in {delay}s... (attempt {retries+1}/{MAX_RETRIES})")
            else:
                print(f"  [error] Copilot API error (attempt {retries+1}/{MAX_RETRIES}): {e}")
            last_error = e
        finally:
            if session:
                try:
                    await session.destroy()
                except Exception:
                    pass

        retries += 1
        if retries < MAX_RETRIES:
            await asyncio.sleep(delay)
            delay *= 2

    raise Exception(f"Failed to call Copilot after {MAX_RETRIES} retries. Last error: {last_error}")


def _build_prompt(system_prompt: str, messages: list[dict]) -> str:
    """
    Build a single prompt string from system prompt + conversation history.

    The Copilot SDK session.send_and_wait() takes a single prompt string.
    We format the conversation history into the prompt so the model sees
    the full context.
    """
    parts = [system_prompt.strip(), ""]

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role in ("user",):
            parts.append(content)
            parts.append("")
        elif role in ("model", "assistant"):
            parts.append(f"Assistant: {content}")
            parts.append("")

    parts.append("Now respond with either a search query (search: <query>) or your final answer (Explanation/Exact Answer/Confidence):")

    return "\n".join(parts)
