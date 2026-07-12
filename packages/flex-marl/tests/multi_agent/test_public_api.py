def test_multi_agent_public_api_is_available() -> None:
    from flex_marl import (
        CentralizedOutput,
        FlatFieldConfig,
        MultiAgentEncoderConfig,
        MultiAgentEncoderModule,
        MultiAgentMode,
        SequentialFieldConfig,
    )

    assert all(
        (
            CentralizedOutput,
            FlatFieldConfig,
            MultiAgentEncoderConfig,
            MultiAgentEncoderModule,
            MultiAgentMode,
            SequentialFieldConfig,
        )
    )
