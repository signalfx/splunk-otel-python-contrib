# Overview

This document describes how to release packages from this repository.

## Releasing GenAI Packages

How to release a new version of the GenAI packages
(util-genai, util-genai-evals, genai-emitters-splunk,
genai-evals-deepeval, instrumentation-langchain, genai-translator-traceloop):

### Package Information

| Package Name | Path | Tag Format                         |
|-------------|------|------------------------------------|
| splunk-otel-util-genai | `util/opentelemetry-util-genai` | `util-genai-v0.Y.Z`                |
| splunk-otel-util-genai-evals | `util/opentelemetry-util-genai-evals` | `util-genai-evals-v0.Y.Z`          |
| splunk-otel-genai-emitters-splunk | `util/opentelemetry-util-genai-emitters-splunk` | `genai-emitters-splunk-v0.Y.Z`     |
| splunk-otel-genai-evals-deepeval | `util/opentelemetry-util-genai-evals-deepeval` | `genai-evals-deepeval-v0.Y.Z`      |
| splunk-otel-instrumentation-langchain | `instrumentation-genai/opentelemetry-instrumentation-langchain` | `instrumentation-langchain-v0.Y.Z` |
| splunk-otel-genai-translator-traceloop | `util/opentelemetry-util-genai-translator-traceloop` | `genai-translator-traceloop-v0.Y.Z` |

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
    - **IMPORTANT**: GenAI packages use a tag format that includes the package name prefix
    - Examples:
        - For `splunk-otel-util-genai` version `0.1.0`, create tag: `util-genai-v0.1.0`
        - For `splunk-otel-instrumentation-langchain` version `0.2.0`, create tag: `instrumentation-langchain-v0.2.0`
    - A tag matching the package-specific format will trigger the CI pipeline to build and publish that specific package to PyPI

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
