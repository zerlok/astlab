from pytest_case_provider.mark import FeatureFlagMark, VersionRange

FEATURE_UNION_TYPE_SYNTAX = FeatureFlagMark("union type syntax", VersionRange.python((3, 10)))

FEATURE_TYPE_ALIAS_QUALNAME = FeatureFlagMark("type alias qualname", VersionRange.python((3, 11)))

FEATURE_TYPE_ALIAS_SYNTAX = FeatureFlagMark("type alias syntax", VersionRange.python((3, 12)))

FEATURE_TYPE_VAR_SYNTAX = FeatureFlagMark("type var syntax", VersionRange.python((3, 12)))

FEATURE_FORWARD_REF = FeatureFlagMark("forward references", VersionRange.python((3, 14)))

# See: https://docs.python.org/3/whatsnew/3.14.html#whatsnew314-typing-union
FEATURE_TYPING_UNION_IS_UNION_TYPE = FeatureFlagMark(
    "repr(typing.Optional[T]) == 'T | None'",
    VersionRange.python((3, 14)),
)
