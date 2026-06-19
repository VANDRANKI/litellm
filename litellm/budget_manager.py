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
from typing import Dict, Literal, List, Optional, Union

import litellm
from litellm.constants import (
    DAYS_IN_A_MONTH,
    DAYS_IN_A_WEEK,
    DAYS_IN_A_YEAR,
    HOURS_IN_A_DAY,
)
from litellm.utils import ModelResponse


class BudgetManager:
    """Manages per-user LLM spend budgets with optional time-based resets.

    Tracks cumulative spend for each user against a configurable total budget.
    Budgets can be scoped to daily, weekly, monthly, or yearly durations and
    are automatically reset when the period expires.

    Storage is either local (JSON file on disk) or hosted (remote LiteLLM API).

    Args:
        project_name: Identifier for the project; used as a namespace when
            data is stored in the hosted backend.
        client_type: ``"local"`` (default) stores budgets in ``user_cost.json``
            in the current working directory. ``"hosted"`` uses the LiteLLM
            API specified by ``api_base``.
        api_base: Base URL for the hosted budget API.
            Defaults to ``"https://api.litellm.ai"``.
        headers: HTTP headers to send with every hosted API request.
            Defaults to ``{"Content-Type": "application/json"}``.

    Example:
        >>> manager = BudgetManager(project_name="my-app")
        >>> manager.create_budget(total_budget=10.0, user="alice", duration="monthly")
        >>> manager.update_cost(user="alice", completion_obj=response)
        >>> if manager.get_current_cost("alice") > manager.get_total_budget("alice"):
        ...     raise Exception("Budget exceeded")
    """

    def __init__(
        self,
        project_name: str,
        client_type: str = "local",
        api_base: Optional[str] = None,
        headers: Optional[dict] = None,
    ):
        self.client_type = client_type
        self.project_name = project_name
        self.api_base = api_base or "https://api.litellm.ai"
        self.headers = headers or {"Content-Type": "application/json"}
        ## load the data or init the initial dictionaries
        self.load_data()

    def print_verbose(self, print_statement: str) -> None:
        """Log a message at DEBUG level when verbose mode is enabled.

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

        For ``"local"`` storage, reads ``user_cost.json`` from the current
        working directory (initializes an empty dict if the file does not exist).
        For ``"hosted"`` storage, fetches the user dict from the remote API.

        Populates ``self.user_dict`` in-place.
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
    ) -> dict:
        """Register a spend budget for a user.

        Creates or replaces the budget entry for *user*. If *duration* is
        provided the budget will be automatically reset each period.

        Args:
            total_budget: Maximum allowed spend in USD for the period.
            user: Unique user identifier.
            duration: Optional reset cadence.  One of ``"daily"``,
                ``"weekly"``, ``"monthly"``, or ``"yearly"``.  When ``None``
                the budget never resets automatically.
            created_at: Unix timestamp marking the start of the current budget
                period. Defaults to the current time.

        Returns:
            The newly created budget entry dict for *user*.

        Raises:
            ValueError: If *duration* is not one of the accepted literals.
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
        """Estimate cost of a completion request *before* sending it.

        Counts prompt tokens from *messages* and adds the user's existing
        spend to give a projected total cost if the request were sent.

        Args:
            model: LiteLLM model string (e.g. ``"gpt-4o"``).
            messages: List of chat message dicts in OpenAI format.
            user: User whose current cost is factored into the projection.

        Returns:
            Projected total cost in USD (current cost + estimated prompt cost).
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
        """Return the total budget limit for *user* in USD.

        Args:
            user: Unique user identifier.

        Returns:
            The configured total budget in USD.

        Raises:
            KeyError: If *user* does not exist in the budget manager.
        """
        return self.user_dict[user]["total_budget"]

    def update_cost(
        self,
        user: str,
        completion_obj: Optional[ModelResponse] = None,
        model: Optional[str] = None,
        input_text: Optional[str] = None,
        output_text: Optional[str] = None,
    ) -> dict:
        """Record the cost of a completed LLM request for *user*.

        Accepts either a ``ModelResponse`` object (preferred — cost is computed
        automatically) **or** explicit ``model`` + ``input_text`` +
        ``output_text`` strings (cost is estimated from token counts).

        Args:
            user: Unique user identifier.
            completion_obj: A ``litellm.ModelResponse`` returned by a
                completion call.  When provided, ``model``, ``input_text``,
                and ``output_text`` are ignored.
            model: LiteLLM model string.  Required when *completion_obj* is
                not provided.
            input_text: The prompt text sent to the model.  Required when
                *completion_obj* is not provided.
            output_text: The completion text returned by the model.  Required
                when *completion_obj* is not provided.

        Returns:
            A dict ``{"user": <updated budget entry>}``.

        Raises:
            ValueError: If neither *completion_obj* nor all three text
                arguments are provided.
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
        """Return the current accumulated spend for *user* in USD.

        Args:
            user: Unique user identifier.

        Returns:
            Current cost in USD, or ``0`` if no spend has been recorded yet.
        """
        return self.user_dict[user].get("current_cost", 0)

    def get_model_cost(self, user: str) -> Union[Dict[str, float], int]:
        """Return per-model cost breakdown for *user*.

        Args:
            user: Unique user identifier.

        Returns:
            A dict mapping model name to cumulative cost in USD, or ``0`` if
            no model-level spend has been recorded.
        """
        return self.user_dict[user].get("model_cost", 0)

    def is_valid_user(self, user: str) -> bool:
        """Check whether *user* has a registered budget entry.

        Args:
            user: Unique user identifier.

        Returns:
            ``True`` if the user exists in the budget manager, ``False``
            otherwise.
        """
        return user in self.user_dict

    def get_users(self) -> List[str]:
        """Return a list of all user identifiers with registered budgets.

        Returns:
            List of user identifier strings.
        """
        return list(self.user_dict.keys())

    def reset_cost(self, user: str) -> dict:
        """Reset *user*'s current spend to zero.

        Clears both ``current_cost`` and the per-model cost breakdown.
        Does **not** modify the ``total_budget`` or duration settings.

        Args:
            user: Unique user identifier.

        Returns:
            A dict ``{"user": <updated budget entry>}``.
        """
        self.user_dict[user]["current_cost"] = 0
        self.user_dict[user]["model_cost"] = {}
        return {"user": self.user_dict[user]}

    def reset_on_duration(self, user: str) -> None:
        """Reset *user*'s cost if the configured budget duration has elapsed.

        Compares the current time against ``last_updated_at`` + the configured
        duration. If the period has expired, calls :meth:`reset_cost` and
        updates ``last_updated_at`` to now.

        Args:
            user: Unique user identifier.  Must have ``duration`` and
                ``last_updated_at`` keys in their budget entry.
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
        """Check and reset budgets for every user whose duration period has expired.

        Iterates over all registered users and calls :meth:`reset_on_duration`
        for any user that has a configured ``duration``.  Safe to call
        periodically (e.g., in a background thread or cron job).
        """
        for user in self.get_users():
            if "duration" in self.user_dict[user]:
                self.reset_on_duration(user)

    def _save_data_thread(self) -> None:
        """Persist budget data to storage in a non-blocking background thread."""
        thread = threading.Thread(
            target=self.save_data
        )  # [Non-Blocking]: saves data without blocking execution
        thread.start()

    def save_data(self) -> dict:
        """Persist the current budget data to the configured storage backend.

        For ``"local"`` storage, writes ``user_cost.json`` in the current
        working directory (pretty-printed with 4-space indentation).
        For ``"hosted"`` storage, sends the user dict to the remote API.

        Returns:
            For local storage: ``{"status": "success"}``.
            For hosted storage: the JSON response body from the remote API.
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
