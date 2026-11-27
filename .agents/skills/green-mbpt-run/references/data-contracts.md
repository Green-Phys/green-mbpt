# Data contracts

Treat GREEN HDF5 files as versioned interfaces, not informal containers.

- Input contract: <https://green-phys.org/docs/components/data-formats/input-file/>
- Output contract: <https://green-phys.org/docs/components/data-formats/output-files/>
- Data-format index and dimension notation:
  <https://green-phys.org/docs/components/data-formats/>

Never guess dataset dimensions, spin layout, symmetry expansion, units, or the
meaning of an iteration marker. Preserve and check the documented version
attributes. A file being readable by HDF5 does not make it compatible with the
selected Green release.

For an incompatible legacy file, follow the migration command reported by the
selected executable or the current documentation. Do not manually synthesize
new version attributes or symmetry metadata to bypass validation.

An interrupted calculation can leave partially written state. Confirm the
documented restart contract before treating an iteration marker as a complete
checkpoint.

