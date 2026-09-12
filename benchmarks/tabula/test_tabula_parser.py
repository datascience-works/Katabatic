import pandas as pd

from katabatic.models.tabula.utils import _convert_text_to_tabular_data


def test_tabula_parser_reconstructs_valid_row():
    columns = [
        "age",
        "workclass",
        "education",
        "education-num",
        "occupation",
        "relationship",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "class",
    ]

    valid_row = (
        "age 33, workclass Self-emp-not-inc, education 12th, "
        "education-num 8, occupation Craft-repair, relationship Husband, "
        "capital-gain 0, capital-loss 0, hours-per-week 32, "
        "native-country United-States, class <=50K"
    )

    df_gen = pd.DataFrame(columns=columns)
    df_gen = _convert_text_to_tabular_data([valid_row], df_gen)

    assert len(df_gen) == 1
    assert not df_gen.isna().any(axis=1).iloc[0]
