import flex_marl
import flex_marl.multi_agent as multi_agent


PUBLIC_NAMES = {
    "CentralizedOutput",
    "FieldConfig",
    "FlatFieldConfig",
    "MultiAgentEncoderConfig",
    "MultiAgentEncoderModule",
    "MultiAgentMode",
    "SequentialFieldOptions",
    "SequentialFieldConfig",
    "validate_field_config",
    "validate_multi_agent_encoder_config",
    "validate_sequential_field_options",
}


def test_multi_agent_namespace_exports_documented_api() -> None:
    assert PUBLIC_NAMES == set(multi_agent.__all__)
    assert all(getattr(multi_agent, name) is not None for name in PUBLIC_NAMES)


def test_root_namespace_exports_documented_multi_agent_api() -> None:
    assert PUBLIC_NAMES.issubset(flex_marl.__all__)
    assert all(getattr(flex_marl, name) is getattr(multi_agent, name) for name in PUBLIC_NAMES)


def test_internal_compilation_helpers_are_not_root_exports() -> None:
    assert "compile_head_config" not in flex_marl.__all__
    assert "_internal_key" not in flex_marl.__all__
    assert not hasattr(flex_marl, "compile_head_config")
    assert not hasattr(flex_marl, "_internal_key")


def test_enum_values_match_serializable_configuration_strings() -> None:
    assert {member.value for member in multi_agent.MultiAgentMode} == {"shared", "independent", "centralized"}
    assert {member.value for member in multi_agent.CentralizedOutput} == {"global", "broadcast"}
