from types import SimpleNamespace

from d_atc.environment import visualization


def test_read_only_client_blocks_forwarded_simulation_commands() -> None:
    messages: list[str] = []
    clientstack = SimpleNamespace(
        forward=lambda *args, **kwargs: None,
        echo=messages.append,
    )
    stack_module = SimpleNamespace(forward=lambda *args, **kwargs: None)
    stackbase = SimpleNamespace(forward=lambda *args, **kwargs: None)

    visualization._install_read_only_forward(clientstack, stack_module, stackbase)
    clientstack.forward("HDG AC1,180")
    stack_module.forward()
    stackbase.forward("RESET")

    assert messages == [
        visualization._READ_ONLY_MESSAGE,
        visualization._READ_ONLY_MESSAGE,
        visualization._READ_ONLY_MESSAGE,
    ]


def test_stack_receiver_ignores_empty_network_frames() -> None:
    received: list[str] = []
    receiver = visualization._safe_stack_receiver(received.append)

    receiver()
    receiver("")
    receiver("PAN 1,2")

    assert received == ["PAN 1,2"]
