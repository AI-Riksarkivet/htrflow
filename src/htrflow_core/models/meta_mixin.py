class MetadataMixin:
    def default_metadata(self):
        """Define default metadata values."""
        return {"framework": self.get_default_framework(), "task": self.get_default_task(), "device": self.device}

    @property
    def task(self):
        return self.metadata["task"]

    @property
    def framework(self):
        return self.metadata["framework"]

    def get_default_framework(self):
        """Provide a default framework name."""
        return "Generic_framework"  # Should be overridden

    def get_default_task(self):
        """Provide a default task name."""
        return "Generic_task"  # Should be overridden
