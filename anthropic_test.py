from anthropic import anthropic


def assert_equals(actual, expected, message=None):
    """
    Assert that two objects are equal. Raise an AssertionError with a detailed message if not.
    """
    assert actual == expected, message or f"Expected:\n{expected}\nActual:\n{actual}"


# 1. System message handling
def test_system_message_handling():
    input_data = {
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ],
    }

    expected = {
        "system": "You are helpful",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 4096,
        "model": None
    }

    assert_equals(anthropic(input_data), expected)


# 2. Basic message conversion
def test_basic_message_conversion():
    input_data = {
        "messages": [{"role": "user", "content": "Hello"}],
    }

    expected = {
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 4096,
        "model": None
    }

    assert_equals(anthropic(input_data), expected)


# 2b. Multimodal content handling
def test_multimodal_content_handling():
    input_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,abc123"},
                    },
                ],
            },
        ],
    }

    expected = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "abc123",
                        },
                    },
                ],

            },
        ],
        "max_tokens": 4096,
        "model": None
    }

    assert_equals(anthropic(input_data), expected)


# 3. Parameter conversion
def test_parameter_conversion():
    input_data = {
        "messages": [{"role": "user", "content": "Hi"}],
        "model": "claude-3-5-sonnet-20241002",
        "max_tokens": 100,
        "metadata": {"user_id": "123"},
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "stop": ["END"],
    }

    expected = {
        "messages": [{"role": "user", "content": "Hi"}],
        "model": "claude-3-5-sonnet-20241002",
        "max_tokens": 100,
        "metadata": {"user_id": "123"},
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "stop_sequences": ["END"],

    }

    assert_equals(anthropic(input_data), expected)


# 3b. Array stop sequences
def test_array_stop_sequences():
    input_data = {
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": ["STOP", "END"],
    }

    expected = {
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 4096,
        "stop_sequences": ["STOP", "END"],
        "model": None
    }

    assert_equals(anthropic(input_data), expected)


# 4. Tool handling
def test_tool_handling():
    input_data = {
        "messages": [{"role": "user", "content": "Hi"}],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "tools": [
            {
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            },
        ],
    }

    expected = {
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 4096,
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": False},
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather info",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        ],
        "model": None
    }

    assert_equals(anthropic(input_data), expected)


# 4b. Specific tool choice
def test_specific_tool_choice():
    input_data = {
        "messages": [{"role": "user", "content": "Hi"}],
        "tool_choice": {"function": {"name": "get_weather"}},
    }

    expected = {
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 4096,
        "tool_choice": {"type": "tool", "name": "get_weather"},
        "model": None
    }

    assert_equals(anthropic(input_data), expected)
