class Metric:
    """
    The base class of metrics.
    @param is_accumulated: bool, if the metric value is accumulated. For example, recall and precision are accumulated
        values, but iou is not accumulated.
    """

    def __init__(self, is_accumulated:bool):
        self.is_accumulated = is_accumulated

    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)

    def zero_values(self):
        pass

    def forward(self, output, target):
        assert 0, "must implement this function!"

    @property
    def accumulated(self):
        return self.is_accumulated
