# Releasing Packages

This document describes how to release packages from this repository.

## Table of Contents
- [Releasing splunk-opentelemetry](#releasing-splunk-opentelemetry)
- [Releasing GenAI Packages](#releasing-genai-packages)

---

## Releasing splunk-opentelemetry

How to release a new version of the `splunk-opentelemetry` project:

1) Create a new branch from `main`
2) Bump dependency versions in pyproject.toml
    - update otel dependencies to the latest versions, e.g.:
        - `"opentelemetry-exporter-otlp-proto-http==1.36.0"`
        - `"opentelemetry-instrumentation==0.57b0"`
3) Bump our version in __about__.py
4) Update additional version string locations
    - `ott_lib.py` # this file is used as a library for integration tests
    - `docker/requirements.txt` # this file is used to build the docker init image for the operator
    - `docker/example-instrumentation.yaml`  # this file is just an example but would be nice to show the latest version
5) Add a new entry in CHANGELOG.md
6) Commit the changes with a message like "Bump version to 3.4.5"
    - you may want to use multiple commits for clarity, e.g.:
        - bump dependency versions
        - bump our version in `__about__.py`
        - update additional version string locations
        - update CHANGELOG.md
7) Push the changes to the Github Splunk OTel Python repo
8) Open a PR and merge after approval
9) Navigate to the GitLab mirror and verify that the mirror has pulled the version you just merged by checking the
   version number in the `__about__.py` file
10) When ready to release, create a new tag like `v3.4.5` on main in GitLab
    - a tag of the format `vX.Y.Z` will trigger the CI pipeline to build and publish the package to PyPI and the Docker
      image to Quay
11) Monitor the release pipeline in GitLab to ensure it completes successfully
12) Post release, verify that the new package is available on PyPI and the Docker image is available on Quay
13) Smoke test the release locally by installing the new package and running it with a small app
14) Navigate to Pipelines in the GitLab repo, click the download button for the build job that just ran,
    and select the 'build-job' artifact
    - this will download a tarball of the package files
15) Navigate to the Splunk OTel Python repo and create a New Release
    - create a new tag on publish with the tag name you created in step 10
    - set the title to that tag name (e.g. `v2.7.0`)
    - unpack the tarball from step 14 and drag its contents onto the attachments section of the New Release page
    - Leave the defaults selected and click Publish

---

## Releasing GenAI Packages

How to release a new version of the GenAI packages (util-genai, util-genai-evals, genai-emitters-splunk, genai-evals-deepeval, instrumentation-langchain):

### Package Information

| Package Name | Path | Tag Format |
|-------------|------|------------|
| splunk-otel-util-genai | `util/opentelemetry-util-genai` | `util-genai-vX.Y.Z` |
| splunk-otel-util-genai-evals | `util/opentelemetry-util-genai-evals` | `util-genai-evals-vX.Y.Z` |
| splunk-otel-genai-emitters-splunk | `util/opentelemetry-util-genai-emitters-splunk` | `genai-emitters-splunk-vX.Y.Z` |
| splunk-otel-genai-evals-deepeval | `util/opentelemetry-util-genai-evals-deepeval` | `genai-evals-deepeval-vX.Y.Z` |
| splunk-otel-instrumentation-langchain | `instrumentation-genai/opentelemetry-instrumentation-langchain` | `instrumentation-langchain-vX.Y.Z` |

### Release Steps

1) Create a new branch from `main`

2) Navigate to the package directory you want to release

3) Bump dependency versions in the package's `pyproject.toml`
    - update otel dependencies to the latest versions if needed

4) Bump the version in the package's `version.py` file
    - e.g., `util/opentelemetry-util-genai/src/opentelemetry/util/genai/version.py`
    - update `__version__ = "0.1.0"` to the new version

5) Add a new entry in CHANGELOG.md for the package release

6) Commit the changes with a message like "Bump util-genai version to 0.1.0"

7) Push the changes to the Github Splunk OTel Python repo

8) Open a PR and merge after approval

9) Navigate to the GitLab mirror and verify that the mirror has pulled the version you just merged by checking the
   version number in the package's `version.py` file

10) When ready to release, create a new tag with the **package-specific format** on main in GitLab
    - **IMPORTANT**: Unlike the main package, GenAI packages use a tag format that includes the package name prefix
    - Examples:
        - For `splunk-otel-util-genai` version `0.1.0`, create tag: `util-genai-v0.1.0`
        - For `splunk-otel-instrumentation-langchain` version `0.2.0`, create tag: `instrumentation-langchain-v0.2.0`
    - A tag matching the package-specific format will trigger the CI pipeline to build and publish that specific package to PyPI
    - Optionally, Docker images will be published to Quay (for packages configured with Docker publishing)
    
    **Helper Script**: You can use the helper script to create tags:
    ```bash
    ./release-test-scripts/create-release-tag.sh util-genai 0.1.0
    ```

11) Monitor the release pipeline in GitLab to ensure it completes successfully
    - The pipeline will show jobs like `build-util-genai`, `sign-util-genai`, `deploy-util-genai`

12) Post release, verify that the new package is available on PyPI
    - e.g., `https://pypi.org/project/splunk-otel-util-genai/0.1.0/`

13) Smoke test the release locally by installing the new package and running it with a small app
    ```bash
    pip install splunk-otel-util-genai==0.1.0
    ```

14) Navigate to Pipelines in the GitLab repo, click the download button for the build job that just ran,
    and select the package-specific build job artifact (e.g., `build-util-genai`)
    - this will download a tarball of the package files

15) Navigate to the Splunk OTel Python repo and create a New Release
    - create a new tag on publish with the tag name you created in step 10 (e.g., `util-genai-v0.1.0`)
    - set the title to that tag name
    - unpack the tarball from step 14 and drag its contents onto the attachments section of the New Release page
    - Leave the defaults selected and click Publish

### Key Differences from Main Package Release

- **Tag Format**: GenAI packages use `{package-name}-vX.Y.Z` instead of just `vX.Y.Z`
- **Version Location**: Each package has its own `version.py` file in its source directory
- **Independent Releases**: Each package can be released independently without affecting others
- **Pipeline Jobs**: Each package has dedicated build/sign/deploy jobs in GitLab CI

### Additional Resources

- See `RELEASE_GENAI_PACKAGES.md` for detailed release documentation
- See `REFACTORING_SUMMARY.md` for technical details about the CI/CD setup
