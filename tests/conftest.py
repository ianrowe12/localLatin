"""Test-session setup for the research-side suite.

sqlite3 is imported here, before pytest collects anything, purely for import
order. On the HPC conda env (`localLatin`, Python 3.10) importing pandas first
loads a libstdc++ that cannot satisfy the CXXABI version conda's libicui18n
wants, and the C extension behind sqlite3 then fails to load:

    ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.15' not found
                 (required by .../libicui18n.so.78)

Imported the other way round both work. Collection is alphabetical, so
test_bulk_attribution_guards.py (pandas) is imported before
test_provision_smoke_account.py (sqlite3) and the suite becomes uncollectable on
the cluster even though every individual module is fine. CI does not hit this at
all, since it runs a clean setup-python rather than the conda env.

This is a workaround for a broken environment, not a statement about the code.
"""

import sqlite3  # noqa: F401  (import for side effect: load the C extension first)
