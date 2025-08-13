# Component tests

Focused, lightweight tests for modular components. Suites covered:
- attention
- sampling
- decomposition
- encoder
- decoder
- output_heads (added)
- losses (added)

All tests are marked `@pytest.mark.extended` and use tiny tensors. They validate:
- registry presence and uniqueness
- minimal instantiation via factory/registry
- forward pass shapes and gradient flow (when applicable)
- small behavior checks where cheap (e.g., quantile shapes)
