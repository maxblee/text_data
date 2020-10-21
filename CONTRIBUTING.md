# Contributing

Contributions are always welcome and will always be credited.

## Types of Contributions

Please suggest all contributions at [https://www.github.com/maxblee/text_data/issues/](https://www.github.com/maxblee/text_data/issues/). 

### Report Bugs

If you are reporting a bug, please specify:

- Your operating system name and version (assuming you think it might be relevant)
- Your version of Python (assuming you think it might be relevant)
- A detailed list of steps required to replicate the issue.
- Any other details that might be helpful.

Bugs will be labeled with the "Type: Bug" tag.

### Documentation Fixes

This project could always use better documentation. If the existing documentation
is confusing or if you think it could be expanded, please provide the following information:

- The place in the documentation that you find confusing, insufficient, or incorrect. (Please include a link to the page if the documentation problem lies with the documentation page. Please include a link to the line number for docstring changes.)
- What you think needs to be fixed.

Documentation issues will be labeled with the "Type: Documentation" tag.

### Features

To propose features, please:

- Explain the feature in detail
- Keep the scope as narrow as possible

Features will be labeled with the "Type: Enhancement" tag.

### Other Features

Feel free to file any other issue, question, or proposal you have.

## Implementing Contributions

In addition to filing issues, pull requests would be appreciated. Anything with the "Status: Available" or "Status: Help Wanted" tags is available for whoever wants it.

## Pull Request Guidelines

Before you submit a pull request, please ensure that the following are true:

- It contains tests.
- It contains documentation.
- All of the tests pass using `make test-suite`

## Deploying (for Maintainers)

To deploy, first edit `CHANGELOG.md` to reflect new changes. Then, update the version in `pyproject.toml`.
Then type

```bash
git push
git push --tags
```

The versioning is in the format `MAJOR.MINOR.PATCH`.
Git tags are in the form `vMAJOR.MINOR.PATCH`.

This will deploy to PyPI if the tests pass.