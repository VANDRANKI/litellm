# +-----------------------------------------------+
# |                                               |
# |           NOT PROXY BUDGET MANAGER            |
# |  proxy budget manager is in proxy_server.py   |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import json
import os
import threading
import time
from typing import Any, Dict, Literal, Optional

import litellm
from litellm.constants import (
    DAYS_IN_A_MONTH,
    DAYS_IN_A_WEEK,
    DAYS_IN_A_YEAR,
    HOURS_IN_A_DAY,
)
from litellm.utils import ModelResponse


class BudgetManager:
    """Manages per-user LLM spend budgets with optional duration-based resets.

    Supports two storage backends:
    - ``"local"``: persists budget data to a ``user_cost.json`` file in the
      current working directory.
    - ``"hosted"``: reads from and writes to the LiteLLM hosted API at
      ``api_base``.

    Example usage::

        manager = BudgetManager(project_name="my-project")
        manager.create_budget(total_budget=10.0, user="alice", duration="monthly")

        # After each LLM call:
        manager.update_cost(user="alice", completion_obj=response)

        remaining = manager.get_total_budget("alice") - manager.get_current_cost("alice")
    """

    def __init__(
        self,
        project_name: str,
        client_type: str = "local",
        api_base: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize the BudgetManager.

        Args:
            project_name: Identifier for the project, used when syncing with the
                hosted backend.
            client_type: Storage backend to use. Either ``"local"`` (default) or
                ``"hosted"``.
            api_base: Base URL for the hosted LiteLLM API. Defaults to
                ``"https://api.litellm.ai"`` when ``client_type="hosted"``.
            headers: HTTP headers to include in hosted API requests. Defaults to
                ``{"Content-Type": "application/json"}``.
        """
        self.client_type = client_type
        self.project_name = project_name
        self.api_base = api_base or "https://api.litellm.ai"
        self.headers = headers or {"Content-Type": "application/json"}
        ## load the data or init the initial dictionaries
        self.user_dict: Dict[str, Any] = {}
        self.load_data()

    def print_verbose(self, print_statement: str) -> None:
        """Log a message when verbose mode is enabled.

        Args:
            print_statement: The message to log.
        """
        try:
            if litellm.set_verbose:
                import logging

                logging.info(print_statement)
        except Exception:
            pass

    def load_data(self) -> None:
        """Load budget data from the configured storage backend.

        For ``"local"`` storage this reads ``user_cost.json`` from the current
        directory (initializing an empty dict when the file is absent). For
        ``"hosted"`` storage this fetches data from the LiteLLM API.
        """
        if self.client_type == "local":
            # Check if user dict file exists
            if os.path.isfile("user_cost.json"):
                # Load the user dict
                with open("user_cost.json", "r") as json_file:
                    self.user_dict = json.load(json_file)
            else:
                self.print_verbose("User Dictionary not found!")
                self.user_dict = {}
            self.print_verbose(f"user dict from local: {self.user_dict}")
        elif self.client_type == "hosted":
            # Load the user_dict from hosted db
            url = self.api_base + "/get_budget"
            data = {"project_name": self.project_name}
            response = litellm.module_level_client.post(
                url, headers=self.headers, json=data
            )
            response = response.json()
            if response["status"] == "error":
                self.user_dict = (
                    {}
                )  # assume this means the user dict hasn't been stored yet
            else:
                self.user_dict = response["data"]

    def create_budget(
        self,
        total_budget: float,
        user: str,
        duration: Optional[Literal["daily", "weekly", "monthly", "yearly"]] = None,
        created_at: float = time.time(),
    ) -> Dict[str, Any]:
        """Create or overwrite a budget entry for a user.

        Args:
            total_budget: Maximum allowed spend in USD for this user.
            user: Unique identifier for the user.
            duration: Optional rolling reset period. When set, the budget resets
                automatically after the specified interval. One of ``"daily"``,
                ``"weekly"``, ``"monthly"``, or ``"yearly"``. When ``None``
                (default) the budget never resets automatically.
            created_at: Unix timestamp marking when the budget period starts.
                Defaults to the current time.

        Returns:
            The newly created budget record for the user.

        Raises:
            ValueError: If ``duration`` is not one of the accepted string
                literals.
        """
        self.user_dict[user] = {"total_budget": total_budget}
        if duration is None:
            return self.user_dict[user]

        if duration == "daily":
            duration_in_days = 1
        elif duration == "weekly":
            duration_in_days = DAYS_IN_A_WEEK
        elif duration == "monthly":
            duration_in_days = DAYS_IN_A_MONTH
        elif duration == "yearly":
            duration_in_days = DAYS_IN_A_YEAR
        else:
            raise ValueError(
                """duration needs to be one of ["daily", "weekly", "monthly", "yearly"]"""
            )
        self.user_dict[user] = {
            "total_budget": total_budget,
            "duration": duration_in_days,
            "created_at": created_at,
            "last_updated_at": created_at,
        }
        self._save_data_thread()  # [Non-Blocking] Update persistent storage without blocking execution
        return self.user_dict[user]

    def projected_cost(self, model: str, messages: list, user: str) -> float:
        """Estimate the total cost if the given messages were sent now.

        Counts the prompt tokens of ``messages``, calculates their cost, and
        adds the user's current cumulative spend to produce a projected total.

        Args:
            model: The LiteLLM model string (e.g. ``"openai/gpt-4o"``).
            messages: A list of chat messages in OpenAI format.
            user: The user identifier whose current spend is included.

        Returns:
            Projected total cost in USD (current spend + estimated prompt cost).
        """
        text = "".join(message["content"] for message in messages)
        prompt_tokens = litellm.token_counter(model=model, text=text)
        prompt_cost, _ = litellm.cost_per_token(
            model=model, prompt_tokens=prompt_tokens, completion_tokens=0
        )
        current_cost = self.user_dict[user].get("current_cost", 0)
        projected_cost = prompt_cost + current_cost
        return projected_cost

    def get_total_budget(self, user: str) -> float:
        """Return the configured total budget for a user.

        Args:
            user: The user identifier to look up.

        Returns:
            The total budget in USD as set via :meth:`create_budget`.
        """
        return self.user_dict[user]["total_budget"]

    def update_cost(
        self,
        user: str,
        completion_obj: Optional[ModelResponse] = None,
        model: Optional[str] = None,
        input_text: Optional[str] = None,
        output_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record the cost of an LLM call against a user's budget.

        You must supply either ``completion_obj`` **or** all three of ``model``,
        ``input_text``, and ``output_text``.

        Args:
            user: The user identifier to charge.
            completion_obj: A :class:`~litellm.utils.ModelResponse` returned by
                :func:`litellm.completion`. When provided, cost is derived from
                its token usage metadata.
            model: The LiteLLM model string. Required when ``completion_obj`` is
                not supplied.
            input_text: The prompt text sent to the model. Required when
                ``completion_obj`` is not supplied.
            output_text: The completion text returned by the model. Required
                when ``completion_obj`` is not supplied.

        Returns:
            A dict with a single key ``"user"`` containing the updated budget
            record.

        Raises:
            ValueError: If neither ``completion_obj`` nor the text/model triple
                is provided.
        """
        if model and input_text and output_text:
            prompt_tokens = litellm.token_counter(
                model=model, messages=[{"role": "user", "content": input_text}]
            )
            completion_tokens = litellm.token_counter(
                model=model, messages=[{"role": "user", "content": output_text}]
            )
            (
                prompt_tokens_cost_usd_dollar,
                completion_tokens_cost_usd_dollar,
            ) = litellm.cost_per_token(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            cost = prompt_tokens_cost_usd_dollar + completion_tokens_cost_usd_dollar
        elif completion_obj:
            cost = litellm.completion_cost(completion_response=completion_obj)
            model = completion_obj[
                "model"
            ]  # if this throws an error try, model = completion_obj['model']
        else:
            raise ValueError(
                "Either a chat completion object or the text response needs to be passed in. Learn more - https://docs.litellm.ai/docs/budget_manager"
            )

        self.user_dict[user]["current_cost"] = cost + self.user_dict[user].get(
            "current_cost", 0
        )
        if "model_cost" in self.user_dict[user]:
            self.user_dict[user]["model_cost"][model] = cost + self.user_dict[user][
                "model_cost"
            ].get(model, 0)
        else:
            self.user_dict[user]["model_cost"] = {model: cost}

        self._save_data_thread()  # [Non-Blocking] Update persistent storage without blocking execution
        return {"user": self.user_dict[user]}

    def get_current_cost(self, user: str) -> float:
        """Return the total spend accumulated so far for a user.

        Args:
            user: The user identifier to look up.

        Returns:
            Current cumulative cost in USD, or ``0`` if no spend has been
            recorded yet.
        """
        return self.user_dict[user].get("current_cost", 0)

    def get_model_cost(self, user: str) -> Dict[str, float]:
        """Return the per-model cost breakdown for a user.

        Args:
            user: The user identifier to look up.

        Returns:
            A dict mapping model name to total USD spend on that model, or
            ``0`` if no per-model data exists yet.
        """
        return self.user_dict[user].get("model_cost", 0)

    def is_valid_user(self, user: str) -> bool:
        """Check whether a budget entry exists for the given user.

        Args:
            user: The user identifier to check.

        Returns:
            ``True`` if the user has a budget record, ``False`` otherwise.
        """
        return user in self.user_dict

    def get_users(self) -> list:
        """Return all user identifiers that have budget records.

        Returns:
            A list of user identifier strings.
        """
        return list(self.user_dict.keys())

    def reset_cost(self, user: str) -> Dict[str, Any]:
        """Reset the tracked spend for a user back to zero.

        Clears both the aggregate ``current_cost`` and the per-model
        ``model_cost`` breakdown stored for the given user.

        Args:
            user: The user identifier whose cost should be reset.

        Returns:
            dict: ``{"user": <updated user record>}`` reflecting the reset state.
        """
        self.user_dict[user]["current_cost"] = 0
        self.user_dict[user]["model_cost"] = {}
        return {"user": self.user_dict[user]}

    def reset_on_duration(self, user: str) -> None:
        """Reset a user's cost if their budget duration period has elapsed.

        Compares the current time against ``last_updated_at`` plus the
        configured ``duration`` (in days). If the period has passed, calls
        :meth:`reset_cost` and updates ``last_updated_at`` to the current time,
        then persists the change asynchronously.

        Args:
            user: The user identifier to evaluate.
        """
        # Get current and creation time
        last_updated_at = self.user_dict[user]["last_updated_at"]
        current_time = time.time()

        # Convert duration from days to seconds
        duration_in_seconds = (
            self.user_dict[user]["duration"] * HOURS_IN_A_DAY * 60 * 60
        )

        # Check if duration has elapsed
        if current_time - last_updated_at >= duration_in_seconds:
            # Reset cost if duration has elapsed and update the creation time
            self.reset_cost(user)
            self.user_dict[user]["last_updated_at"] = current_time
            self._save_data_thread()  # Save the data

    def update_budget_all_users(self) -> None:
        """Apply duration-based resets to all users that have one configured.

        Iterates over every tracked user and calls :meth:`reset_on_duration`
        for users whose budget record includes a ``duration`` field.
        """
        for user in self.get_users():
            if "duration" in self.user_dict[user]:
                self.reset_on_duration(user)

    def _save_data_thread(self) -> None:
        """Persist budget data to storage in a background thread (non-blocking)."""
        thread = threading.Thread(
            target=self.save_data
        )  # [Non-Blocking]: saves data without blocking execution
        thread.start()

    def save_data(self) -> Dict[str, Any]:
        """Persist the in-memory budget dict to the configured storage backend.

        For ``"local"`` storage, writes ``user_cost.json`` in the current
        directory. For ``"hosted"`` storage, POSTs to the LiteLLM API.

        Returns:
            A dict with a ``"status"`` key indicating success or the raw API
            response from the hosted backend.
        """
        if self.client_type == "local":
            import json

            # save the user dict
            with open("user_cost.json", "w") as json_file:
                json.dump(
                    self.user_dict, json_file, indent=4
                )  # Indent for pretty formatting
            return {"status": "success"}
        elif self.client_type == "hosted":
            url = self.api_base + "/set_budget"
            data = {"project_name": self.project_name, "user_dict": self.user_dict}
            response = litellm.module_level_client.post(
                url, headers=self.headers, json=data
            )
            response = response.json()
            return response
        return {"status": "no-op"}
