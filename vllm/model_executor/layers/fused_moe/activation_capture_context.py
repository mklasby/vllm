# START CB
class ActivationCaptureContext:
    """A simple thread-safe context to store captured activations."""

    _instance = None
    _enabled = False

    def __init__(self):
        self.reset()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add_activations(self, activations_tensor, topk_tensor, topk_weights):
        """Adds a cloned tensor of activations to the context."""
        if self._enabled:
            self.captured_activations.append(activations_tensor.clone())
            self.selected_experts.append(topk_tensor.clone())
            self.topk_weights.append(topk_weights.clone())

    def add_router_logits(self, router_logits):
        """Adds the current router logits to the context."""
        if self._enabled:
            self.router_logits.append(router_logits.clone())

    def retrieve_state(self):
        """Retrieves and clears the stored activations."""
        state = dict(
            activations=self.captured_activations,
            chunk_size=self.chunk_size,
            topk_weights=self.topk_weights,
            router_logits=self.router_logits 
        )
        self.reset()
        return state

    def reset(self):
        """Clears any stored activations."""
        self.captured_activations = []
        self.selected_experts = []
        self.topk_weights = []
        self.chunk_size = None
        self.router_logits = []
# END CB