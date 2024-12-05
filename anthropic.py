def anthropic(body):
    """
    Convert an OpenAI body to an Anthropic body
    :param body: dict
    :return: dict
    """
    # System messages are specified at the top level in Anthropic
    system = next((msg for msg in body["messages"] if msg["role"] == "system"), None)

    # Convert messages
    messages = [
        {
            "role": msg["role"],
            "content": [
                {"type": "text", "text": content["text"]} if content["type"] == "text" else
                {"type": "image", "source": anthropic_source_from_url(content["image_url"]["url"])}
                for content in msg["content"]
            ] if isinstance(msg["content"], list) else msg["content"]
        }
        for msg in body["messages"] if msg["role"] != "system"
    ]

    # Parallel tool calls
    parallel_tool_calls = (
        {"disable_parallel_tool_use": not body["parallel_tool_calls"]}
        if isinstance(body.get("parallel_tool_calls"), bool) else {}
    )

    # Map OpenAI parameters to Anthropic equivalents
    params = {
        "model": body.get("model"),
        "max_tokens": body.get("max_tokens", 4096),
        **({"metadata": {"user_id": body["metadata"]["user_id"]}} if body.get("metadata", {}).get("user_id") else {}),
        **({"stream": body["stream"]} if isinstance(body.get("stream"), bool) else {}),
        **({"temperature": body["temperature"]} if isinstance(body.get("temperature"), (int, float)) else {}),
        **({"top_p": body["top_p"]} if isinstance(body.get("top_p"), (int, float)) else {}),
        # Convert single string or list of stop sequences
        **({"stop_sequences": [body["stop"]]} if isinstance(body.get("stop"), str) else
           {"stop_sequences": body["stop"]} if isinstance(body.get("stop"), list) else {}),
        # Convert tool_choice to Anthropic's equivalent
        **(
            {"tool_choice": {"type": "auto", **parallel_tool_calls}}
            if body.get("tool_choice") == "auto" else
            {"tool_choice": {"type": "any", **parallel_tool_calls}}
            if body.get("tool_choice") == "required" else
            {}
            if body.get("tool_choice") == "none" else
            {"tool_choice": {
                "type": "tool",
                "name": body["tool_choice"].get("function", {}).get("name"),
                **parallel_tool_calls
            }} if isinstance(body.get("tool_choice"), dict) else {}
        )
    }

    # Convert function definitions to Anthropic's tool format
    tools = [
        {
            "name": tool["function"]["name"],
            "description": tool["function"]["description"],
            "input_schema": tool["function"]["parameters"],
        }
        for tool in body.get("tools", [])
    ] if body.get("tools") else None

    # Return the constructed Anthropic body
    return {
        **({"system": system["content"]} if system else {}),
        "messages": messages,
        **params,
        **({"tools": tools} if tools else {}),
    }


def anthropic_source_from_url(url):
    """
    Handle data URIs in Anthropic's format. External URLs are not supported.
    :param url: str
    :return: dict
    """
    if url.startswith("data:"):
        base, base64_data = url.split(",", 1)
        return {
            "type": "base64",
            "media_type": base.replace("data:", "").replace(";base64", ""),
            "data": base64_data,
        }
