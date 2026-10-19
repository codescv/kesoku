"""Machine-readable error codes and shared lookup tables."""

from enum import StrEnum


class ErrorCode(StrEnum):
    """Error codes carried by :class:`domerr.errors.DomainError`.

    Every member's value equals its name so codes serialise unchanged into API responses.
    """

    E_DOM_CORE_001 = "E_DOM_CORE_001"  # value has the wrong type
    E_DOM_CORE_002 = "E_DOM_CORE_002"  # required value is missing
    E_DOM_REGISTRY_001 = "E_DOM_REGISTRY_001"  # two validators registered for one field
    E_DOM_REGISTRY_002 = "E_DOM_REGISTRY_002"  # no validator registered for a field
    E_DOM_EMAIL_001 = "E_DOM_EMAIL_001"  # email missing or not a string
    E_DOM_EMAIL_002 = "E_DOM_EMAIL_002"  # email malformed
    E_DOM_PHONE_001 = "E_DOM_PHONE_001"  # phone missing or not a string
    E_DOM_PHONE_002 = "E_DOM_PHONE_002"  # phone not E.164
    E_DOM_ADDRESS_001 = "E_DOM_ADDRESS_001"  # address is not a mapping
    E_DOM_ADDRESS_002 = "E_DOM_ADDRESS_002"  # required address sub-field missing or empty
    E_DOM_ADDRESS_003 = "E_DOM_ADDRESS_003"  # unknown country
    E_DOM_RECORD_001 = "E_DOM_RECORD_001"  # record is not a mapping


# Country names and common spellings seen in intake data, mapped to ISO-3166 alpha-2 codes.
# Canonical codes map to themselves.
COUNTRY_ALIASES: dict[str, str] = {
    "US": "US",
    "USA": "US",
    "U.S.": "US",
    "U.S.A.": "US",
    "United States": "US",
    "United States of America": "US",
    "DE": "DE",
    "Germany": "DE",
    "Deutschland": "DE",
    "GB": "GB",
    "UK": "GB",
    "United Kingdom": "GB",
    "Great Britain": "GB",
    "FR": "FR",
    "France": "FR",
    "JP": "JP",
    "Japan": "JP",
    "Nippon": "JP",
    "CN": "CN",
    "China": "CN",
    "PRC": "CN",
    "CA": "CA",
    "Canada": "CA",
    "NL": "NL",
    "Netherlands": "NL",
    "Holland": "NL",
    "CH": "CH",
    "Switzerland": "CH",
    "Schweiz": "CH",
}
