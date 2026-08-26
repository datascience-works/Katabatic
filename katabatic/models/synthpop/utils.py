"""
Utility functions for SynthPop integration in Katabatic.

Provides helpers for:
- Writing the R script that calls the synthpop package
"""

from pathlib import Path


def write_r_script(
    r_script_path: Path,
    input_csv: Path,
    output_csv: Path,
    seed: int,
) -> None:
    """
    Write an R script that runs synthpop CART-based synthesis.

    The generated script:
    - Loads the synthpop R package
    - Reads the input CSV
    - Runs syn() with CART method
    - Writes the synthetic output CSV

    Args:
        r_script_path: Path where the R script will be saved.
        input_csv: Path to the input training CSV file.
        output_csv: Path where synthetic CSV output will be written.
        seed: Random seed for reproducibility.
    """
    r_code = f"""
suppressMessages(library(synthpop))

set.seed({seed})

data <- read.csv("{input_csv.as_posix()}")

syn_data <- syn(
    data,
    method = "cart",
    seed = {seed}
)

write.csv(
    syn_data$syn,
    "{output_csv.as_posix()}",
    row.names = FALSE
)
"""

    r_script_path.write_text(r_code.strip())
