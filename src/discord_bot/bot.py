"""Discord bot for LITRIS semantic search.

Exposes LITRIS search functionality via Discord slash commands
with embed formatting and button-based pagination.
"""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    import discord
    from discord import app_commands

    HAS_DISCORD = True
except ImportError:
    HAS_DISCORD = False

from src.discord_bot.formatters import (
    RESULTS_PER_PAGE,
    format_paper_embed,
    format_search_page,
    format_summary_embed,
)
from src.mcp.adapters import LitrisAdapter

logger = logging.getLogger(__name__)


# PaginationView must be guarded because discord.ui.View is the base class
# and discord.py may not be installed.
if HAS_DISCORD:

    class PaginationView(discord.ui.View):
        """Button-based pagination for search results."""

        def __init__(
            self,
            results: list[dict[str, Any]],
            query: str,
            timeout: float = 180.0,
        ):
            super().__init__(timeout=timeout)
            self.results = results
            self.query = query
            self.page = 0
            self.total_pages = (len(results) + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE
            self._update_buttons()

        def _update_buttons(self) -> None:
            """Enable/disable buttons based on current page."""
            self.prev_button.disabled = self.page <= 0
            self.next_button.disabled = self.page >= self.total_pages - 1

        def _get_page_embeds(self) -> list[discord.Embed]:
            """Get embeds for the current page."""
            start = self.page * RESULTS_PER_PAGE
            end = start + RESULTS_PER_PAGE
            page_results = self.results[start:end]

            embed_dicts = format_search_page(page_results, self.query, self.page, len(self.results))
            return [discord.Embed.from_dict(d) for d in embed_dicts]

        @discord.ui.button(label="Previous", style=discord.ButtonStyle.secondary)
        async def prev_button(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ) -> None:
            """Go to previous page."""
            self.page = max(0, self.page - 1)
            self._update_buttons()
            await interaction.response.edit_message(embeds=self._get_page_embeds(), view=self)

        @discord.ui.button(label="Next", style=discord.ButtonStyle.primary)
        async def next_button(
            self, interaction: discord.Interaction, button: discord.ui.Button
        ) -> None:
            """Go to next page."""
            self.page = min(self.total_pages - 1, self.page + 1)
            self._update_buttons()
            await interaction.response.edit_message(embeds=self._get_page_embeds(), view=self)


def create_bot(adapter: LitrisAdapter | None = None) -> discord.Client:
    """Create and configure the LITRIS Discord bot.

    Args:
        adapter: Optional pre-configured LitrisAdapter. If None, creates one.

    Returns:
        Configured discord.Client with slash commands.
    """
    if not HAS_DISCORD:
        raise ImportError(
            "discord.py is required for the Discord bot. Install with: pip install discord.py>=2.3"
        )

    if adapter is None:
        adapter = LitrisAdapter()

    intents = discord.Intents.default()
    client = discord.Client(intents=intents)
    tree = app_commands.CommandTree(client)

    @client.event
    async def on_ready() -> None:
        """Sync commands when bot is ready."""
        await tree.sync()
        logger.info(f"LITRIS bot ready as {client.user}")

    @tree.command(name="search", description="Search the LITRIS literature index")
    @app_commands.describe(
        query="Natural language search query",
        top_k="Number of results (default: 10)",
        year_min="Minimum publication year",
        year_max="Maximum publication year",
    )
    async def search_command(
        interaction: discord.Interaction,
        query: str,
        top_k: int = 10,
        year_min: int | None = None,
        year_max: int | None = None,
    ) -> None:
        """Search the literature index."""
        await interaction.response.defer()

        try:
            results = adapter.search(
                query=query,
                top_k=top_k,
                year_min=year_min,
                year_max=year_max,
                include_extraction=True,
            )

            result_list = results.get("results", [])
            if not result_list:
                await interaction.followup.send(f"No results found for: **{query}**")
                return

            if len(result_list) > RESULTS_PER_PAGE:
                view = PaginationView(result_list, query)
                embeds = view._get_page_embeds()
                await interaction.followup.send(embeds=embeds, view=view)
            else:
                embed_dicts = format_search_page(result_list, query, 0, len(result_list))
                embeds = [discord.Embed.from_dict(d) for d in embed_dicts]
                await interaction.followup.send(embeds=embeds)

        except Exception as e:
            logger.error(f"Search command failed: {e}")
            await interaction.followup.send(f"Search failed: {e}")

    @tree.command(name="paper", description="Get full details for a paper")
    @app_commands.describe(paper_id="LITRIS paper identifier")
    async def paper_command(
        interaction: discord.Interaction,
        paper_id: str,
    ) -> None:
        """Get paper details."""
        await interaction.response.defer()

        try:
            result = adapter.get_paper(paper_id)

            if not result.get("found"):
                await interaction.followup.send(f"Paper not found: **{paper_id}**")
                return

            embed_dict = format_paper_embed(result)
            embed = discord.Embed.from_dict(embed_dict)
            await interaction.followup.send(embed=embed)

        except Exception as e:
            logger.error(f"Paper command failed: {e}")
            await interaction.followup.send(f"Paper lookup failed: {e}")

    @tree.command(name="similar", description="Find papers similar to a given paper")
    @app_commands.describe(
        paper_id="Source paper identifier",
        top_k="Number of similar papers (default: 5)",
    )
    async def similar_command(
        interaction: discord.Interaction,
        paper_id: str,
        top_k: int = 5,
    ) -> None:
        """Find similar papers."""
        await interaction.response.defer()

        try:
            results = adapter.find_similar(paper_id, top_k=top_k)

            if not results.get("result_count"):
                await interaction.followup.send(f"No similar papers found for: **{paper_id}**")
                return

            source_title = results.get("source_title", paper_id)
            similar = results.get("similar_papers", [])

            # Reformat for search page display
            formatted = []
            for s in similar:
                formatted.append(
                    {
                        "title": s.get("title", "Unknown"),
                        "authors": s.get("authors", ""),
                        "year": s.get("year"),
                        "score": s.get("score", 0),
                        "paper_id": s.get("paper_id", ""),
                        "matched_text": "",
                        "extraction": s.get("extraction", {}),
                    }
                )

            if len(formatted) > RESULTS_PER_PAGE:
                view = PaginationView(formatted, f"Similar to: {source_title}")
                embeds = view._get_page_embeds()
                await interaction.followup.send(embeds=embeds, view=view)
            else:
                embed_dicts = format_search_page(
                    formatted,
                    f"Similar to: {source_title}",
                    0,
                    len(formatted),
                )
                embeds = [discord.Embed.from_dict(d) for d in embed_dicts]
                await interaction.followup.send(embeds=embeds)

        except Exception as e:
            logger.error(f"Similar command failed: {e}")
            await interaction.followup.send(f"Similar search failed: {e}")

    @tree.command(name="summary", description="Get LITRIS index statistics")
    async def summary_command(interaction: discord.Interaction) -> None:
        """Get index summary."""
        await interaction.response.defer()

        try:
            summary = adapter.get_summary()
            embed_dict = format_summary_embed(summary)
            embed = discord.Embed.from_dict(embed_dict)
            await interaction.followup.send(embed=embed)

        except Exception as e:
            logger.error(f"Summary command failed: {e}")
            await interaction.followup.send(f"Summary failed: {e}")

    # Store references for testing
    client._tree = tree
    client._adapter = adapter

    return client


def run_bot(token: str | None = None) -> None:
    """Run the LITRIS Discord bot.

    Args:
        token: Discord bot token. If None, reads from DISCORD_BOT_TOKEN env var.
    """
    if not HAS_DISCORD:
        raise ImportError("discord.py is required. Install with: pip install discord.py>=2.3")

    if token is None:
        token = os.environ.get("DISCORD_BOT_TOKEN")

    if not token:
        raise ValueError(
            "Discord bot token required. Set DISCORD_BOT_TOKEN environment variable "
            "or pass token directly."
        )

    client = create_bot()
    client.run(token, log_handler=None)
