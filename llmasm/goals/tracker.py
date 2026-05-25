"""Goal lifecycle helper."""

from __future__ import annotations

from llmasm.graph.models import Goal
from llmasm.ids import new_id
from llmasm.storage.base import Storage


class GoalTracker:
    """Thin goal lifecycle wrapper over storage."""

    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def load_active_goal(self, workspace_graph_id: str) -> Goal | None:
        """Load the active goal."""

        return self.storage.load_active_goal(workspace_graph_id)

    def create_provisional_goal(self, workspace_graph_id: str, prompt: str) -> Goal:
        """Create a provisional goal before planning."""

        goal = Goal(id=new_id("goal"), workspace_graph_id=workspace_graph_id, text=prompt)
        self.storage.persist_goal(goal)
        return goal

    def finalize_goal(
        self, goal_id: str, goal_update_text: str, active_task_graph_id: str | None = None
    ) -> Goal:
        """Finalize a provisional goal after accepted compilation."""

        goal = self.storage.load_goal(goal_id)
        goal.text = goal_update_text or goal.text
        goal.status = "active"
        goal.active_task_graph_id = active_task_graph_id
        self.storage.update_goal(goal)
        return goal

    def steer_goal(self, goal_id: str, goal_update_text: str) -> Goal:
        """Update an existing active goal."""

        goal = self.storage.load_goal(goal_id)
        goal.text = goal_update_text or goal.text
        goal.status = "active"
        self.storage.update_goal(goal)
        return goal

    def close_goal(self, goal_id: str, reason: str) -> Goal:
        """Close a goal."""

        goal = self.storage.load_goal(goal_id)
        goal.status = "closed"
        goal.metadata["close_reason"] = reason
        self.storage.update_goal(goal)
        return goal
