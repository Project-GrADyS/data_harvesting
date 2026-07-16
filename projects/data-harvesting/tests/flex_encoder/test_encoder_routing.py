import torch
from torch import nn

from data_harvesting.encoding import _TensorDictEncoderRouter


class _CapturingEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return inputs["agent_mask"].unsqueeze(-1).to(torch.float32)


def test_router_passes_environment_masks_without_deriving_or_changing_them() -> None:
    encoder = _CapturingEncoder()
    router = _TensorDictEncoderRouter(encoder)
    # Identical numeric values can be padding or valid data; only the mask decides.
    sensors = torch.tensor([[[[-1.0, -1.0], [-1.0, -1.0]]]])
    sensors_mask = torch.tensor([[[False, True]]])
    drones_mask = torch.tensor([[[True, False]]])
    agent_mask = torch.tensor([[True]])

    router(
        sensors=sensors,
        sensors_mask=sensors_mask,
        drones_mask=drones_mask,
        agent_mask=agent_mask,
    )

    assert encoder.inputs["sensors"] is sensors
    assert encoder.inputs["sensors_mask"] is sensors_mask
    assert encoder.inputs["drones_mask"] is drones_mask
    assert encoder.inputs["agent_mask"] is agent_mask
    assert encoder.inputs["sensors_mask"].tolist() == [[[False, True]]]
