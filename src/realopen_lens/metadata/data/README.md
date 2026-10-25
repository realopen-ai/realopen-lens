# Offline city index

GeoNames cities15000 and countryInfo, downloaded 2026-10-25.
Source: https://download.geonames.org/export/dump/
Attribution: GeoNames (https://www.geonames.org/).
License: Creative Commons Attribution 4.0 (https://creativecommons.org/licenses/by/4.0/).

Changes: selected city IDs/names/countries/admin codes/coordinates/populations and
alternate names; normalized JSON, deterministically gzip-compressed. No personal
photo locations are included in this dataset. Geographic data is provided as-is.

Cities: 34188; gzip bytes: 3078832.
Source ZIP SHA256: 9c5e0684a0aeaa1b8a62ec08f2edcea94a425626806b8b249c1d9fc9b09168ec
CountryInfo SHA256: c55f638cfaafb4fe45a9aa15120a0f1f78e8a34a8e1847d7b9320cf740fbc9ab
Packaged data SHA256: 800af0cb70f6453f8c96a10e4382ddf239858feb4e29deae4a4a4d406eecc00e

Rebuild using scripts/build_places.py with those two source files. The build tool
reads local files only. Lens reads the bundled dataset at runtime; it does not use
GeoNames' web service. City queries use a displayed radius, not administrative
boundaries. Duplicate city names need country/region clarification unless indexed
GPS points identify exactly one candidate. Country-only geofencing is not provided.
