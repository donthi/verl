import litellm
import json # Still needed for error parsing potentially, and response construction
import asyncio
import time
import uuid # For generating unique IDs if needed for litellm ModelResponse
from typing import List, Dict, Any, Optional, AsyncGenerator

# Import fastapi_poe and necessary types
import fastapi_poe as fp
from fastapi_poe.types import ProtocolMessage, BotResponseChunk, ErrorResponse # Add other types if used

from litellm.llms.base import BaseLLM
from litellm.utils import ModelResponse, Choices, Message, Delta, Usage, StreamingChunk
from litellm.exceptions import APIError, AuthenticationError, BadRequestError, ServiceUnavailableError, TimeoutError, RateLimitError


class PoeAdapter(BaseLLM):
    api_key: str
    # Optional: Store other configurations like base_url for Poe API if needed to override fastapi_poe default
    # poe_base_url: Optional[str] = None

    def __init__(self, api_key: str, **kwargs):
        super().__init__()
        if not api_key: # Ensure api_key is not None or empty
             raise AuthenticationError("Poe API key not provided or empty. Set it in the `api_key` argument.")
        self.api_key = api_key
        # Default timeout for get_bot_response is 60s. Can make this configurable.
        self.request_timeout = kwargs.get("request_timeout", 60.0)
        # self.poe_base_url = kwargs.get("poe_base_url") # If allowing override of fastapi_poe's default base URL

    # _validate_environment is no longer strictly necessary as api_key is checked in __init__
    # but can be kept if other environment checks are added later.
    def _validate_environment(self) -> None:
        if not self.api_key:
            raise AuthenticationError("Poe API key is missing.")

    def _get_actual_model_name(self, model: str) -> str:
        # This helper remains useful for litellm's "poe/model_name" convention
        # litellm model name might be "poe/model-name"
        if model.startswith("poe/"):
            return model.split("/", 1)[1]
        return model

    def _transform_messages_to_poe_protocol(self, messages: List[Dict[str, str]]) -> List[ProtocolMessage]:
        poe_messages: List[ProtocolMessage] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            # Map litellm roles to Poe ProtocolMessage roles
            # fastapi_poe.types. ProtocolMessageRole = Literal["system", "user", "bot"]
            if role == "assistant":
                poe_role = "bot"
            elif role == "user":
                poe_role = "user"
            elif role == "system":
                poe_role = "system"
            else:
                raise BadRequestError(f"Unknown role: {role} in messages. Must be 'system', 'user', or 'assistant'.")

            # Attachments are not handled in this basic version but could be added
            # by checking `msg.get("attachments")` and using `fp.upload_file`.
            poe_messages.append(ProtocolMessage(role=poe_role, content=content))
        return poe_messages

    async def completion(self,
                         model: str,
                         messages: List[Dict[str, str]],
                         **kwargs: Any) -> ModelResponse:
        """
        Process a non-streaming completion request with the Poe API
        by using `fastapi_poe.get_bot_response` and accumulating results.
        """
        self._validate_environment() # Checks if api_key is set
        actual_model_name = self._get_actual_model_name(model)
        poe_protocol_messages = self._transform_messages_to_poe_protocol(messages)

        # Arguments for fp.get_bot_response
        # (messages, bot_name, api_key, session, base_url, timeout_s, temperature, skip_system_prompt, user_id, conversation_id, message_id, client_nonce, logit_bias)
        api_kwargs = {
            "temperature": kwargs.get("temperature"),
            "top_p": kwargs.get("top_p"), # Add top_p as it's in fp.get_bot_response signature
            "skip_system_prompt": kwargs.get("skip_system_prompt"), # boolean
            "timeout_s": float(kwargs.get("request_timeout", self.request_timeout)), # Ensure it's float
            "user_id": kwargs.get("user"), # litellm often passes 'user' kwarg
            "conversation_id": kwargs.get("conversation_id"), # Allow passing conversation_id
            "message_id": kwargs.get("message_id"), # Allow passing message_id
            "logit_bias": kwargs.get("logit_bias"),
            # "base_url": self.poe_base_url, # If user wants to override default https://api.poe.com/bot/
        }
        # Filter out None values to pass only explicitly set parameters
        api_kwargs_filtered = {k: v for k, v in api_kwargs.items() if v is not None}

        full_response_text = ""
        # Use a consistent ID generation scheme for the response
        response_id = f"poe-cmpl-{str(uuid.uuid4())}"
        created_time = int(time.time())
        final_chunk_metadata = {} # To store any metadata from the last relevant chunk

        try:
            async for chunk in fp.get_bot_response(
                messages=poe_protocol_messages,
                bot_name=actual_model_name,
                api_key=self.api_key,
                **api_kwargs_filtered
            ):
                # chunk is a BotResponseChunk
                # BotResponseChunk(text, raw_response, suggested_replies, error, meta, is_suggested_reply, is_replace_response, message_id, created_at, request_id, usage_data)
                if chunk.error:
                    error_text = chunk.error.text
                    error_type = chunk.error.error_type
                    error_code = chunk.error.error_code # Can be None

                    # More refined error mapping based on Poe's potential error types
                    if error_type == "authentication_error" or "auth" in error_text.lower() or error_code == 401:
                        raise AuthenticationError(f"Poe API Authentication Error: {error_text}")
                    elif error_type == "rate_limit_error" or error_code == 429:
                        raise RateLimitError(f"Poe API Rate Limit Error: {error_text}")
                    elif error_type == "server_error" or (error_code and error_code >= 500):
                        raise ServiceUnavailableError(f"Poe API Server Error: {error_text}")
                    elif error_type == "invalid_request_error" or (error_code and 400 <= error_code < 500 and error_code != 401 and error_code != 429):
                         raise BadRequestError(f"Poe API Invalid Request Error: {error_text}")
                    else: # Generic APIError
                        raise APIError(f"Poe API Error: {error_text}", response_obj=chunk.error, status_code=error_code)

                full_response_text += chunk.text

                # Store metadata from the last chunk that might be useful (e.g. message_id)
                if chunk.message_id: final_chunk_metadata["message_id"] = chunk.message_id
                if chunk.request_id: final_chunk_metadata["request_id"] = chunk.request_id
                # Add raw response of the last chunk for debugging if needed
                # final_chunk_metadata["raw_chunk"] = chunk.raw_response

            # If response_id wasn't set from a chunk, keep the generated one
            if "message_id" in final_chunk_metadata:
                response_id = final_chunk_metadata["message_id"]

            # Poe API (via fastapi_poe client) does not provide token usage in BotResponseChunk directly.
            # BotResponseChunk has `usage_data: Optional[Dict[str, Any]] = None`
            # If `chunk.usage_data` were populated, we could try to map it. For now, assume 0.
            usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            # if final_chunk_metadata.get("usage_data"):
            #    usage.prompt_tokens = final_chunk_metadata["usage_data"].get("prompt_token_count", 0)
            #    usage.completion_tokens = final_chunk_metadata["usage_data"].get("completion_token_count", 0)
            #    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

            model_response = ModelResponse(
                id=response_id,
                choices=[
                    Choices(
                        finish_reason="stop",
                        index=0,
                        message=Message(content=full_response_text, role="assistant")
                    )
                ],
                created=created_time,
                model=model, # Original litellm model name
                usage=usage,
            )
            model_response._hidden_params["poe_metadata"] = final_chunk_metadata
            return model_response

        except fp.client.BotError as e:
            # This is a specific exception from fastapi_poe client library for bot-related errors
            # Example: if the bot_name is invalid.
            raise APIError(f"Poe BotError (completion): {str(e)}")
        except asyncio.TimeoutError:
            timeout_val = api_kwargs_filtered.get('timeout_s', self.request_timeout)
            raise TimeoutError(f"Poe API request timed out after {timeout_val}s.")
        except Exception as e: # Catch any other unexpected errors
            if isinstance(e, litellm.exceptions.LiteLLMException): # Re-raise if already a litellm exception
                raise
            # Consider mapping httpx exceptions if they leak through fastapi_poe
            # e.g. import httpx; if isinstance(e, httpx.ConnectError): raise ServiceUnavailableError(...)
            raise APIError(f"An unexpected error occurred with Poe API (completion): {str(e)}")


    async def async_streaming(self,
                         model: str,
                         messages: List[Dict[str, str]],
                         **kwargs: Any) -> AsyncGenerator[StreamingChunk, None]:
        """
        Process a streaming completion request with the Poe API.

        Yields Server-Sent Events (SSEs) transformed into litellm StreamingChunk objects.

        Args:
            model (str): The model identifier (e.g., "poe/gemini-pro").
            messages (List[Dict[str, str]]): A list of message dictionaries.
            **kwargs: Additional keyword arguments for the API call.

        Yields:
            StreamingChunk: Chunks of the response as they are received.

        Raises:
            AuthenticationError: If the API key is invalid or missing.
            BadRequestError: If the request is malformed.
            RateLimitError: If the API rate limit is exceeded or a message limit is hit.
            ServiceUnavailableError: If the Poe API is unavailable or there's a connection issue.
            TimeoutError: If the request times out.
            APIError: For other API-related errors, including errors sent within the stream.
        """
        """
        Process a streaming completion request with the Poe API
        by using `fastapi_poe.get_bot_response` and yielding transformed chunks.
        """
        self._validate_environment()
        actual_model_name = self._get_actual_model_name(model)
        poe_protocol_messages = self._transform_messages_to_poe_protocol(messages)

        # Arguments for fp.get_bot_response, similar to non-streaming version
        api_kwargs = {
            "temperature": kwargs.get("temperature"),
            "top_p": kwargs.get("top_p"), # Add top_p here as well
            "skip_system_prompt": kwargs.get("skip_system_prompt"),
            "timeout_s": float(kwargs.get("request_timeout", self.request_timeout)),
            "user_id": kwargs.get("user"),
            "conversation_id": kwargs.get("conversation_id"),
            "message_id": kwargs.get("message_id"),
            "logit_bias": kwargs.get("logit_bias"),
            # "base_url": self.poe_base_url,
        }
        api_kwargs_filtered = {k: v for k, v in api_kwargs.items() if v is not None}

        chunk_id_prefix = f"poe-strchunk-{str(uuid.uuid4())}"
        first_content_chunk_processed = False # Tracks if we've sent the first delta with content and role

        try:
            async for i, poe_chunk in enumerate(fp.get_bot_response(
                messages=poe_protocol_messages,
                bot_name=actual_model_name,
                api_key=self.api_key,
                **api_kwargs_filtered
            )):
                # poe_chunk is BotResponseChunk
                if poe_chunk.error:
                    error_text = poe_chunk.error.text
                    error_type = poe_chunk.error.error_type
                    error_code = poe_chunk.error.error_code
                    if error_type == "authentication_error" or "auth" in error_text.lower() or error_code == 401:
                        raise AuthenticationError(f"Poe API Authentication Error (Streaming): {error_text}")
                    elif error_type == "rate_limit_error" or error_code == 429:
                        raise RateLimitError(f"Poe API Rate Limit Error (Streaming): {error_text}")
                    elif error_type == "server_error" or (error_code and error_code >= 500):
                        raise ServiceUnavailableError(f"Poe API Server Error (Streaming): {error_text}")
                    elif error_type == "invalid_request_error" or (error_code and 400 <= error_code < 500 and error_code != 401 and error_code != 429):
                         raise BadRequestError(f"Poe API Invalid Request Error (Streaming): {error_text}")
                    else:
                        raise APIError(f"Poe API Error (Streaming): {error_text}", response_obj=poe_chunk.error, status_code=error_code)

                delta_content = poe_chunk.text
                delta_obj = {}

                # Only include content if it's non-empty.
                # An empty string for content is valid in delta if other things like role/finish_reason are set.
                if delta_content: # If there's text content
                    delta_obj["content"] = delta_content
                    if not first_content_chunk_processed:
                        delta_obj["role"] = "assistant"
                        first_content_chunk_processed = True
                elif not first_content_chunk_processed:
                    # This is an initial chunk without text content.
                    # It might be a metadata chunk, or simply the stream starting.
                    # We need to send the role if this is the very first yielded chunk,
                    # even if content is empty, to establish the assistant's turn.
                    # However, litellm usually expects content or finish_reason with role.
                    # Let's send role only when there's actual content or it's the final chunk.
                    # If this chunk has no text and it's not the first contentful one,
                    # and no other metadata to convey, we might skip it.
                    # Poe's BotResponseChunk has is_replace_response, suggested_replies, meta.
                    # These are not directly translated to litellm's StreamingChunk delta.
                    # So, if no text, and not first content, skip unless it's a special case later.
                    if not poe_chunk.is_replace_response and not poe_chunk.suggested_replies and not poe_chunk.meta:
                        # If it's truly empty and not the first that would carry content, skip.
                        # This avoids sending empty chunks unless they are significant (like final one).
                        # However, if this is the *only* kind of chunk before actual text,
                        # and `first_content_chunk_processed` is still False, we might need to send role.
                        # This logic gets tricky. Simplest: send role with first non-empty text.
                        # If all chunks are empty text, the final chunk will handle the role.
                        if not delta_content: # Explicitly checking again
                            continue # Skip purely empty intermediate chunks

                current_chunk_id = f"{chunk_id_prefix}-{i}"
                litellm_streaming_chunk = StreamingChunk(
                    id=current_chunk_id,
                    choices=[Delta(delta=delta_obj, finish_reason=None, index=0)],
                    created=int(time.time()),
                    model=model
                )
                yield litellm_streaming_chunk

            # After the loop, yield the final chunk with finish_reason="stop"
            final_delta = {}
            if not first_content_chunk_processed: # If no content chunks were ever sent
                 final_delta["role"] = "assistant"
                                     # Ensures role is sent if the response was entirely empty but successful.

            yield StreamingChunk(
                id=f"{chunk_id_prefix}-final",
                choices=[Delta(delta=final_delta, finish_reason="stop", index=0)],
                created=int(time.time()),
                model=model
            )

        except fp.client.BotError as e:
            raise APIError(f"Poe BotError (Streaming): {str(e)}")
        except asyncio.TimeoutError:
            timeout_val = api_kwargs_filtered.get('timeout_s', self.request_timeout)
            raise TimeoutError(f"Poe API stream request timed out after {timeout_val}s.")
        except Exception as e:
            if isinstance(e, litellm.exceptions.LiteLLMException):
                raise
            raise APIError(f"An unexpected error occurred with Poe API (Streaming): {str(e)}")


    # `astream_completion` as requested by prompt, calls litellm standard `async_streaming`
    async def astream_completion(self, model: str, messages: List[Dict[str, str]], **kwargs: Any) -> AsyncGenerator[StreamingChunk, None]:
        async for chunk in self.async_streaming(model, messages, **kwargs):
            yield chunk

    # Embedding method - not requested but good to have as a placeholder
    async def embedding(self, model: str, input: list, **kwargs):
        """
        Placeholder for embedding generation. Not currently supported by this Poe adapter.
        """
        self._validate_environment() # technically, self.api_key existence is checked at __init__
        raise NotImplementedError("Poe adapter does not currently support embeddings.")

# Removed global close_poe_adapter_session() as session is managed by fastapi-poe library internally per call.
