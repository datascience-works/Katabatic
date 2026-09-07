# Evaluation and Performance Benchmarking Data Catalogue

## Adult

This dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult).

| Property           | Value                                              |
| ------------------ | -------------------------------------------------- |
| **Task**           | Binary classification                              |
| **Instances**      | 48,842                                             |
| **Features**       | 14 categorical and integer features                |
| **Target**         | `income`                                           |
| **Classes**        | `<=50K`, `>50K`                                    |
| **Missing values** | Yes                                                |
| **License**        | CC BY 4.0                                          |
| **DOI**            | [10.24432/C5XW20](https://doi.org/10.24432/C5XW20) |

### Fields

| Field            | Type        | Description                                 |
| ---------------- | ----------- | ------------------------------------------- |
| `age`            | Integer     | Age                                         |
| `workclass`      | Categorical | Employment type                             |
| `fnlwgt`         | Integer     | Census final weight                         |
| `education`      | Categorical | Highest education level                     |
| `education-num`  | Integer     | Numerical representation of education level |
| `marital-status` | Categorical | Marital status                              |
| `occupation`     | Categorical | Occupation category                         |
| `relationship`   | Categorical | Family/household relationship               |
| `race`           | Categorical | Race category                               |
| `sex`            | Binary      | `Female`, `Male`                            |
| `capital-gain`   | Integer     | Capital gains                               |
| `capital-loss`   | Integer     | Capital losses                              |
| `hours-per-week` | Integer     | Hours worked per week                       |
| `native-country` | Categorical | Country of origin                           |
| `income`         | Target      | `<=50K` or `>50K`                           |

> **Note:** Missing values occur in `workclass`, `occupation`, and `native-country`.

### Citation

> Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

## Car
This dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/19/car+evaluation).

| Property           | Value                                              |
| ------------------ | -------------------------------------------------- |
| **Task**           | Classification                                     |
| **Instances**      | 1,728                                              |
| **Features**       | 6 categorical features                             |
| **Target**         | `class`                                            |
| **Classes**        | 4                                                  |
| **Missing values** | None                                               |
| **License**        | CC BY 4.0                                          |
| **DOI**            | [10.24432/C5JP48](https://doi.org/10.24432/C5JP48) |

### Fields

| Field      | Description        | Type          |
| ---------- | ------------------ | ------------- |
| `buying`   | Buying price       | Categorical   |
| `maint`    | Maintenance price  | Categorical   |
| `doors`    | Number of doors    | Categorical   |
| `persons`  | Passenger capacity | Categorical   |
| `lug_boot` | Luggage boot size  | Categorical   |
| `safety`   | Estimated safety   | Categorical   |
| `class`    | Car acceptability  | Categorical   |

### Citation

> Bohanec, M. (1988). *Car Evaluation* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5JP48


## Magic

This dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope).

| Property           | Value                                              |
| ------------------ | -------------------------------------------------- |
| **Task**           | Binary classification                              |
| **Instances**      | 19,020                                             |
| **Features**       | 10 continuous features                             |
| **Target**         | `class`                                            |
| **Classes**        | 2                                                  |
| **Missing values** | None                                               |
| **License**        | CC BY 4.0                                          |
| **DOI**            | [10.24432/C52C8B](https://doi.org/10.24432/C52C8B) |

### Fields

| Field      | Description                                                      | Type            |
| ---------- | ---------------------------------------------------------------- | --------------  |
| `fLength`  | Major axis of ellipse                                            | Continuous      |
| `fWidth`   | Minor axis of ellipse                                            | Continuous      |
| `fSize`    | Log of the sum of content of all pixels                          | Continuous      |
| `fConc`    | Ratio of sum of two highest pixels over `fSize`                  | Continuous      |
| `fConc1`   | Ratio of highest pixel over `fSize`                              | Continuous      |
| `fAsym`    | Distance from highest pixel to centre, projected onto major axis | Continuous      |
| `fM3Long`  | Third root of third moment along major axis                      | Continuous      |
| `fM3Trans` | Third root of third moment along minor axis                      | Continuous      |
| `fAlpha`   | Angle of major axis with vector to origin                        | Continuous      |
| `fDist`    | Distance from origin to centre of ellipse                        | Continuous      |
| `class`    | Particle class: `g` = gamma (signal), `h` = hadron (background)  | Binary          |

### Citation

> Bock, R. (2004). *MAGIC Gamma Telescope* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52C8B

## Dataset

This dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/76/nursery).

| Property           | Value                                              |
| ------------------ | -------------------------------------------------- |
| **Task**           | Multiclass classification                          |
| **Instances**      | 12,960                                             |
| **Features**       | 8 categorical features                             |
| **Target**         | `class`                                            |
| **Missing values** | None                                               |
| **License**        | CC BY 4.0                                          |
| **DOI**            | [10.24432/C5P88W](https://doi.org/10.24432/C5P88W) |

### Fields

| Field      | Description                    | Type        |
| ---------- | ------------------------------ | ----------- |
| `parents`  | Parents' occupation            | Categorical |
| `has_nurs` | Child's nursery                | Categorical |
| `form`     | Form of the family             | Categorical |
| `children` | Number of children             | Categorical |
| `housing`  | Housing conditions             | Categorical |
| `finance`  | Financial standing             | Categorical |
| `social`   | Social conditions              | Categorical |
| `health`   | Health conditions              | Categorical |
| `class`    | Nursery application evaluation | Categorical |

### Citation

> Rajkovic, V. (1989). *Nursery* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5P88W

## Shuttle

This dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/148/statlog+shuttle).

| Property           | Value                                              |
| ------------------ | -------------------------------------------------- |
| **Task**           | Multiclass classification                          |
| **Instances**      | 58,000                                             |
| **Features**       | 7                                                  |
| **Target**         | `class`                                            |
| **Missing values** | None                                               |
| **License**        | CC BY 4.0                                          |
| **DOI**            | [10.24432/C5WS31](https://doi.org/10.24432/C5WS31) |

### Fields

| Field      | Type    | Description                                        |
| ---------- | ------- | -------------------------------------------------- |
| `Rad Flow` | Integer | Time-related numerical attribute                   |
| `Fpv Close`| Integer | Numerical attribute; meaning not documented by UCI |
| `Fpv Open` | Integer | Numerical attribute; meaning not documented by UCI |
| `High`     | Integer | Numerical attribute; meaning not documented by UCI |
| `Bypass`   | Integer | Numerical attribute; meaning not documented by UCI |
| `Bpv Close`| Integer | Numerical attribute; meaning not documented by UCI |
| `Bpv Open` | Integer | Numerical attribute; meaning not documented by UCI |
| `class`    | Integer | Shuttle operating-state class                      |

### Citation

> *Statlog (Shuttle)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5WS31
