def test_encoder_package_exports_public_config_types():
    from flex_marl.encoder import FlatHeadConfig, PositionalEncodingConfig, SequentialHeadConfig
    assert all((FlatHeadConfig, PositionalEncodingConfig, SequentialHeadConfig))


def test_root_package_exports_public_encoder_api():
    from flex_marl import FlatHeadConfig, MultiHeadEncoderModule, PositionalEncodingConfig, SequentialHeadConfig
    assert all((FlatHeadConfig, MultiHeadEncoderModule, PositionalEncodingConfig, SequentialHeadConfig))


def test_validate_head_config_is_available_from_expected_namespace():
    from flex_marl import validate_head_config
    assert callable(validate_head_config)


def test_internal_head_classes_have_stable_import_path():
    from flex_marl.encoder.heads import FlatHead, SequentialHead
    assert all((FlatHead, SequentialHead))


def test_encoder_package_declares_exact_public_api():
    import flex_marl.encoder as encoder

    assert set(encoder.__all__) == {
        "FlatHeadConfig",
        "PositionalEncodingConfig",
        "SequentialHeadConfig",
        "MultiHeadEncoderModule",
        "validate_head_config",
    }


def test_encoder_configs_use_python_311_syntax():
    import ast
    import inspect
    from flex_marl.encoder import configs

    ast.parse(inspect.getsource(configs), feature_version=(3, 11))
    assert configs.HeadConfig == configs.SequentialHeadConfig | configs.FlatHeadConfig
