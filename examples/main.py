"""Runs an example script, as if it were run directly."""

import runpy

# Options:
# "illustrative_example"
# "flash_unit" (only at the `pre-refactor` tag; see examples/flash_unit/README.md)
example = "illustrative_example"


if __name__ == "__main__":
    runpy.run_module(example, run_name="__main__")
