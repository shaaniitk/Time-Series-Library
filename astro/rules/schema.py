"""Frozen rule-specification dataclasses parsed from declarative JSON.

Rules are data.  Parsing never evaluates expressions, imports modules, or
accepts callables; an operator can only be selected by name from the closed
registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from astro.rules import operators as _operators  # noqa: F401  registers the vocabulary
from astro.rules.naming import validate_emit_key, validate_slug
from astro.rules.registry import RuleContractError, get_operator
from utils.reproducibility import stable_json_hash

RULESET_SCHEMA_VERSION = 1

VALID_SCHOOLS = ("vedic", "western", "both")
VALID_NULL_TYPES = ("date_shift", "body_permute", "phase_randomize")

# Calendar-day half-lives permitted in a response bank, from the feature spec.
ALLOWED_HALF_LIVES = (
    1, 3, 5, 7, 14, 21, 30, 63, 90, 180, 365, 730, 1825, 3650, 7300, 10958,
)


def _require_mapping(value: Any, label: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RuleContractError(f"{label} must be a JSON object, got {type(value).__name__}.")
    return dict(value)


def _reject_unknown_keys(payload: Mapping, allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise RuleContractError(
            f"{label} has unknown keys {unknown}; allowed keys are {sorted(allowed)}."
        )


@dataclass(frozen=True)
class PriorSpec:
    """Declared belief about a rule, used only as a soft, overridable prior."""

    importance: float = 0.5
    l2_anchor: float = 0.0
    l1_weight: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.importance <= 1.0:
            raise RuleContractError(
                f"prior.importance must lie in [0, 1], got {self.importance}."
            )
        if self.l2_anchor < 0.0:
            raise RuleContractError("prior.l2_anchor must be non-negative.")
        if self.l1_weight < 0.0:
            raise RuleContractError("prior.l1_weight must be non-negative.")

    @classmethod
    def from_json(cls, payload: Any) -> "PriorSpec":
        data = _require_mapping(payload, "prior")
        _reject_unknown_keys(data, {"importance", "l2_anchor", "l1_weight"}, "prior")
        return cls(
            importance=float(data.get("importance", 0.5)),
            l2_anchor=float(data.get("l2_anchor", 0.0)),
            l1_weight=float(data.get("l1_weight", 0.0)),
        )

    def to_json(self) -> dict:
        return {
            "importance": self.importance,
            "l2_anchor": self.l2_anchor,
            "l1_weight": self.l1_weight,
        }


@dataclass(frozen=True)
class NullSpec:
    """A matched placebo arm applied to the whole compiled ruleset."""

    type: str
    params: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.type not in VALID_NULL_TYPES:
            raise RuleContractError(
                f"null type must be one of {VALID_NULL_TYPES}, got {self.type!r}."
            )

    @property
    def param_dict(self) -> dict:
        return dict(self.params)

    @classmethod
    def from_json(cls, payload: Any) -> "NullSpec":
        data = _require_mapping(payload, "nulls[]")
        if "type" not in data:
            raise RuleContractError("Each null arm must declare a 'type'.")
        null_type = data.pop("type")
        return cls(type=null_type, params=tuple(sorted(data.items())))

    def to_json(self) -> dict:
        return {"type": self.type, **self.param_dict}

    @property
    def arm_name(self) -> str:
        details = ",".join(f"{key}={value}" for key, value in self.params)
        return f"null:{self.type}:{details}"


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    school: str
    family: str
    operator: str
    inputs: tuple[tuple[str, Any], ...]
    params: tuple[tuple[str, Any], ...]
    emits: tuple[str, ...]
    prior: PriorSpec
    half_lives: tuple[int, ...]
    enabled: bool

    @property
    def input_dict(self) -> dict:
        return dict(self.inputs)

    @property
    def param_dict(self) -> dict:
        return dict(self.params)

    @classmethod
    def from_json(cls, payload: Any) -> "RuleSpec":
        data = _require_mapping(payload, "rules[]")
        _reject_unknown_keys(
            data,
            {
                "rule_id", "school", "family", "operator", "inputs", "params",
                "emits", "prior", "response_bank", "enabled",
            },
            "rule",
        )
        for required in ("rule_id", "school", "family", "operator"):
            if required not in data:
                raise RuleContractError(f"Rule is missing required key {required!r}.")

        rule_id = validate_slug(data["rule_id"], "rule_id")
        family = validate_slug(data["family"], "family")
        school = data["school"]
        if school not in VALID_SCHOOLS:
            raise RuleContractError(
                f"school must be one of {VALID_SCHOOLS}, got {school!r}."
            )

        inputs = _require_mapping(data.get("inputs"), "inputs")
        params = _require_mapping(data.get("params"), "params")

        spec = get_operator(data["operator"])
        spec.validate_call(inputs, params)
        available = tuple(emit.key for emit in spec.resolve_emits(params))

        requested = data.get("emits")
        if requested is None:
            emits = available
        else:
            if not isinstance(requested, list) or not requested:
                raise RuleContractError(
                    f"Rule {rule_id!r}: 'emits' must be a non-empty list of emit keys."
                )
            emits = tuple(validate_emit_key(key) for key in requested)
            unknown = [key for key in emits if key not in available]
            if unknown:
                raise RuleContractError(
                    f"Rule {rule_id!r} requests emits {unknown} which operator "
                    f"{spec.name!r} does not produce. Available: {list(available)}."
                )
            if len(set(emits)) != len(emits):
                raise RuleContractError(f"Rule {rule_id!r} lists duplicate emits.")

        response_bank = _require_mapping(data.get("response_bank"), "response_bank")
        _reject_unknown_keys(response_bank, {"half_life_days"}, "response_bank")
        half_lives = tuple(response_bank.get("half_life_days", ()) or ())
        for half_life in half_lives:
            if half_life not in ALLOWED_HALF_LIVES:
                raise RuleContractError(
                    f"Rule {rule_id!r}: half-life {half_life} is not in the preregistered "
                    f"grid {list(ALLOWED_HALF_LIVES)}."
                )
        if len(set(half_lives)) != len(half_lives):
            raise RuleContractError(f"Rule {rule_id!r} lists duplicate half-lives.")

        return cls(
            rule_id=rule_id,
            school=school,
            family=family,
            operator=spec.name,
            inputs=tuple(sorted(inputs.items())),
            params=tuple(sorted(params.items())),
            emits=emits,
            prior=PriorSpec.from_json(data.get("prior")),
            half_lives=tuple(sorted(half_lives)),
            enabled=bool(data.get("enabled", True)),
        )

    def to_json(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "school": self.school,
            "family": self.family,
            "operator": self.operator,
            "inputs": self.input_dict,
            "params": self.param_dict,
            "emits": list(self.emits),
            "prior": self.prior.to_json(),
            "response_bank": {"half_life_days": list(self.half_lives)},
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class RuleSet:
    ruleset_id: str
    rules: tuple[RuleSpec, ...]
    requires_ephemeris: tuple[tuple[str, Any], ...] = ()
    interaction_pairs: tuple[tuple[str, str], ...] = ()
    null_arms: tuple[NullSpec, ...] = ()
    schema_version: int = RULESET_SCHEMA_VERSION

    @property
    def enabled_rules(self) -> tuple[RuleSpec, ...]:
        return tuple(rule for rule in self.rules if rule.enabled)

    @property
    def families(self) -> tuple[str, ...]:
        seen: list[str] = []
        for rule in self.enabled_rules:
            if rule.family not in seen:
                seen.append(rule.family)
        return tuple(seen)

    @classmethod
    def from_json(cls, payload: Any) -> "RuleSet":
        data = _require_mapping(payload, "ruleset")
        _reject_unknown_keys(
            data,
            {
                "schema_version", "ruleset_id", "rules",
                "requires_ephemeris", "interactions", "null_arms",
            },
            "ruleset",
        )
        version = int(data.get("schema_version", RULESET_SCHEMA_VERSION))
        if version != RULESET_SCHEMA_VERSION:
            raise RuleContractError(
                f"Unsupported ruleset schema_version {version}; expected "
                f"{RULESET_SCHEMA_VERSION}."
            )
        if "ruleset_id" not in data:
            raise RuleContractError("ruleset must declare a 'ruleset_id'.")
        ruleset_id = validate_slug(data["ruleset_id"], "ruleset_id")

        rules_payload = data.get("rules")
        if not isinstance(rules_payload, list) or not rules_payload:
            raise RuleContractError("ruleset must declare a non-empty 'rules' list.")
        rules = tuple(RuleSpec.from_json(item) for item in rules_payload)
        rule_ids = [rule.rule_id for rule in rules]
        if len(set(rule_ids)) != len(rule_ids):
            duplicates = sorted({rid for rid in rule_ids if rule_ids.count(rid) > 1})
            raise RuleContractError(f"Duplicate rule_id values: {duplicates}.")

        interactions = _require_mapping(data.get("interactions"), "interactions")
        _reject_unknown_keys(interactions, {"pairs"}, "interactions")
        pairs_payload = interactions.get("pairs") or []
        pairs: list[tuple[str, str]] = []
        for pair in pairs_payload:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise RuleContractError(
                    "interactions.pairs entries must be [source_channel, target_channel]."
                )
            pairs.append((str(pair[0]), str(pair[1])))
        if pairs:
            raise RuleContractError(
                "interactions.pairs is reserved: the model does not consume declared pairs yet, "
                "and the first confirmatory comparison keeps interaction extensions off."
            )

        requires = _require_mapping(data.get("requires_ephemeris"), "requires_ephemeris")

        null_payload = data.get("null_arms") or []
        if not isinstance(null_payload, list):
            raise RuleContractError("null_arms must be a list.")
        null_arms = tuple(NullSpec.from_json(item) for item in null_payload)
        arm_names = [arm.arm_name for arm in null_arms]
        if len(set(arm_names)) != len(arm_names):
            raise RuleContractError(f"Duplicate null arms: {arm_names}.")

        return cls(
            ruleset_id=ruleset_id,
            rules=rules,
            requires_ephemeris=tuple(sorted(requires.items())),
            interaction_pairs=tuple(pairs),
            null_arms=null_arms,
            schema_version=version,
        )

    def to_json(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "ruleset_id": self.ruleset_id,
            "requires_ephemeris": dict(self.requires_ephemeris),
            "rules": [rule.to_json() for rule in self.rules],
            "interactions": {"pairs": [list(pair) for pair in self.interaction_pairs]},
            "null_arms": [arm.to_json() for arm in self.null_arms],
        }

    @property
    def ruleset_hash(self) -> str:
        return stable_json_hash(self.to_json())


def load_ruleset(path: str) -> RuleSet:
    """Load and validate a ruleset from a JSON file."""

    import json

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return RuleSet.from_json(payload)
