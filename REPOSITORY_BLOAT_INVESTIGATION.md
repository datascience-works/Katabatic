# Repository Bloat Investigation

## Findings

The repository bloat investigation confirmed that most of the repository size is coming from Git history rather than files currently present in the working tree.

The overall repository size was approximately **504 MB**, while the `.git` directory was approximately **473 MB**. Git pack files accounted for around **471.34 MiB**, showing that most of the repository size is stored in historical Git objects.

The largest Git objects included old datasets, archived recordings, compressed files, model checkpoint files, generated data, and old virtual-environment files.

Examples of large historical files identified during the investigation included:

- `Archive/2024-09-02 12-29-48.mkv`
- `Archive/Ganblr-interface/creditcard.csv.zip`
- `Archive/Ganblr-interface/credit_X_train.rar`
- `katabatic/models/codi/CoDi_exp/ckpt.pt`
- `MEG/uploads/drugsComTrain_raw.csv`
- `Paula_CRGAN/AdultT/adult_converted.csv`

A check for files larger than 20 MB in the current working tree showed that the only large file was the Git pack file inside the `.git` directory. This confirms that the main source of repository bloat is historical Git data rather than current project files.

## Recommended Cleanup

The recommended approach is to use `git filter-repo` on a fresh mirror clone, verify the files to be removed, rewrite the Git history, validate the repository size, and then coordinate the updated repository with the team.
