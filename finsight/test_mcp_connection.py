
import asyncio

from mcp import Client
from finsight.mcp_server import mcp


async def main():
    async with Client(mcp) as client:
        tools = await client.list_tools()

        print("Available MCP tools:")
        for tool in tools.tools:
            print("-", tool.name)

        result = await client.call_tool(
            "get_annual_financials",
            {
                "ticker": "NVDA",
                "cik": "0001045810",
                "fiscal_year": 2026,
            },
        )

        if result.is_error:
            raise RuntimeError(f"MCP tool failed: {result.content}")

        print("\nFinancial results:")
        print(result.structured_content or result.content)


if __name__ == "__main__":
    asyncio.run(main())
